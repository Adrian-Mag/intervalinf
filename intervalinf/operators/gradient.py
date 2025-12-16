"""Gradient operator for interval domains."""

import logging
from typing import Optional

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from intervalinf.core.functions import Function


class Gradient(LinearOperator):
    """
    The gradient operator (d/dx) on interval domains.

    In 1D, the gradient is simply the first derivative.
    Uses finite difference method for numerical differentiation.
    """

    def __init__(
        self,
        domain: "Sobolev",
        /,
        *,
        fd_order: int = 2,
        fd_step: Optional[float] = None,
        boundary_treatment: str = 'one_sided'
    ):
        """
        Initialize the gradient operator.

        Args:
            domain: Function space (Lebesgue or SobolevSpace)
            fd_order: Order of finite difference stencil (2, 4)
            fd_step: Step size for finite differences (auto-computed if None)
            boundary_treatment: How to handle boundaries ('one_sided')
        """
        self._domain = domain
        self._codomain = domain
        self._fd_order = fd_order
        self._fd_step = fd_step
        self._boundary_treatment = boundary_treatment
        self._log = logging.getLogger(__name__)

        super().__init__(domain, domain, self._apply)
        self._setup_finite_difference()

    def _setup_finite_difference(self):
        """Setup finite difference method."""
        if self._fd_step is None:
            a = self._domain.function_domain.a
            b = self._domain.function_domain.b
            self._fd_step = (b - a) / 1000

        self._log.debug(
            "GradientOperator (finite difference, order %s) initialized with "
            "step size %.2e",
            self._fd_order,
            self._fd_step,
        )

    def _fd_stencil(self, order: int, location: str = "center"):
        """Return (offsets, coeffs) for first-derivative finite-difference."""
        if order == 2:
            if location == "center":
                return np.array([-1, 1]), np.array([-0.5, 0.5])
            if location == "forward":
                return np.array([0, 1, 2]), np.array([-1.5, 2.0, -0.5])
            if location == "backward":
                return np.array([-2, -1, 0]), np.array([0.5, -2.0, 1.5])
        if order == 4:
            if location == "center":
                offsets = np.array([-2, -1, 1, 2])
                coeffs = np.array([1/12, -2/3, 2/3, -1/12])
                return offsets, coeffs
            if location in ("forward", "backward"):
                return self._fd_stencil(2, location)
        raise ValueError(
            f"Unsupported fd_order={order} or location={location}"
        )

    def _safe_eval(self, func: Function, x: np.ndarray) -> np.ndarray:
        """Evaluate Function `func` on array x safely.
        
        x can be 1D or 2D array. For 2D, evaluates each point.
        """
        try:
            return np.asarray(func(x))
        except Exception:
            # Flatten, evaluate each point, reshape back
            x_flat = np.asarray(x).flatten()
            result = np.asarray([func.evaluate(float(xi)) for xi in x_flat])
            return result.reshape(x.shape)

    def _apply(self, f: Function) -> Function:
        """Apply gradient using finite difference method."""

        def gradient_func(x):
            scalar_input = np.isscalar(x)
            x_arr = np.asarray([x]) if scalar_input else np.asarray(x)

            h = self._fd_step
            a = self._domain.function_domain.a
            b = self._domain.function_domain.b

            y = np.empty_like(x_arr, dtype=float)

            left_mask = x_arr <= a + h
            right_mask = x_arr >= b - h
            interior_mask = ~(left_mask | right_mask)

            # Interior points: central stencil
            if np.any(interior_mask):
                xi = x_arr[interior_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "center")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[interior_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            # Left boundary points: forward stencil
            if np.any(left_mask):
                xl = x_arr[left_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "forward")
                shifts = (xl[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[left_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            # Right boundary points: backward stencil
            if np.any(right_mask):
                xr = x_arr[right_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "backward")
                shifts = (xr[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[right_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            return y[0] if scalar_input else y

        return Function(self._codomain, evaluate_callable=gradient_func)
