"""
Utilities for Robin boundary condition eigenvalue computation.

This module provides a unified implementation of root-finding methods
for computing eigenvalues with Robin boundary conditions.
"""

import math
import numpy as np
from typing import Callable, Tuple


class RobinRootFinder:
    """
    Unified Robin boundary condition eigenvalue computation.

    This class provides methods for finding eigenvalues μ_n that satisfy
    the characteristic equation for Robin boundary conditions:

        D(μ) = (α₀α_L + β₀β_L μ²) sin(μL) + μ(α₀β_L - β₀α_L) cos(μL) = 0

    where:
        - α₀, β₀: left BC coefficients (α₀u(a) + β₀u'(a) = 0)
        - α_L, β_L: right BC coefficients (α_Lu(b) + β_Lu'(b) = 0)
        - L: domain length (b - a)
    """

    @staticmethod
    def bisect(
        F: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-12,
        maxit: int = 100
    ) -> float:
        """
        Standard bisection method for root-finding.

        Args:
            F: Function whose root we seek
            a: Left bracket endpoint
            b: Right bracket endpoint
            tol: Tolerance for convergence
            maxit: Maximum number of iterations

        Returns:
            Approximate root of F in [a, b]

        Raises:
            ValueError: If F(a) and F(b) have the same sign
        """
        fa, fb = F(a), F(b)

        # Check for exact roots at endpoints
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b

        # Ensure sign change
        if fa * fb > 0:
            raise ValueError(
                f"Bisection requires F(a)·F(b) ≤ 0, but "
                f"F({a}) = {fa} and F({b}) = {fb} have the same sign."
            )

        # Bisection iteration
        for _ in range(maxit):
            c = 0.5 * (a + b)
            fc = F(c)

            # Check convergence
            if abs(fc) < tol or 0.5 * (b - a) < tol:
                return c

            # Update bracket
            if fa * fc <= 0.0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        return 0.5 * (a + b)

    @staticmethod
    def find_bracket_with_expansion(
        F: Callable[[float], float],
        left: float,
        right: float,
        max_attempts: int = 6
    ) -> Tuple[float, float]:
        """
        Find a bracketing interval by gentle expansion.

        Args:
            F: Function to bracket
            left: Initial left bound
            right: Initial right bound
            max_attempts: Maximum number of expansion attempts

        Returns:
            Tuple (left, right) where F(left) * F(right) <= 0

        Raises:
            RuntimeError: If bracket cannot be found
        """
        Fl, Fr = F(left), F(right)
        attempts = 0

        while Fl * Fr > 0 and attempts < max_attempts:
            left *= 0.9
            right *= 1.1
            Fl, Fr = F(left), F(right)
            attempts += 1

        if Fl * Fr > 0:
            raise RuntimeError(
                f"Failed to bracket root after {max_attempts} attempts."
            )

        return left, right

    @staticmethod
    def find_bracket_by_scanning(
        F: Callable[[float], float],
        left: float,
        right: float,
        n_samples: int = 129
    ) -> Tuple[float, float]:
        """
        Find a bracketing interval by scanning for sign changes.

        Args:
            F: Function to bracket
            left: Interval left bound
            right: Interval right bound
            n_samples: Number of sample points

        Returns:
            Tuple (left, right) containing a single root

        Raises:
            RuntimeError: If no sign change found
        """
        xs = np.linspace(left, right, n_samples)
        vals = np.array([F(xi) for xi in xs])
        sgn = np.sign(vals)

        idx = np.where(sgn[:-1] * sgn[1:] <= 0)[0]

        if len(idx) == 0:
            raise RuntimeError(
                f"No sign change found in [{left}, {right}]."
            )

        i = idx[0]
        return xs[i], xs[i + 1]

    @staticmethod
    def compute_robin_eigenvalue(
        index: int,
        alpha_0: float,
        beta_0: float,
        alpha_L: float,
        beta_L: float,
        L: float,
        tol: float = 1e-12,
        maxit: int = 100
    ) -> float:
        """
        Compute the k-th eigenvalue μ_k for Robin boundary conditions.

        The eigenvalues satisfy the characteristic equation:
            D(μ) = (α₀α_L + β₀β_L μ²) sin(μL)
                   + μ(α₀β_L - β₀α_L) cos(μL) = 0

        Args:
            index: Which eigenvalue to compute (0-indexed)
            alpha_0, beta_0: Left boundary coefficients
            alpha_L, beta_L: Right boundary coefficients
            L: Domain length
            tol: Root-finding tolerance
            maxit: Maximum bisection iterations

        Returns:
            The eigenvalue μ_index
        """
        # Check for pure Neumann case (α₀ = α_L = 0)
        is_pure_neumann = (abs(alpha_0) < 1e-14 and abs(alpha_L) < 1e-14)

        if is_pure_neumann and index == 0:
            # First eigenvalue for pure Neumann is zero
            return 0.0

        # Define characteristic equation
        def D(mu: float) -> float:
            sin_val = math.sin(mu * L)
            cos_val = math.cos(mu * L)
            return ((alpha_0 * alpha_L + beta_0 * beta_L * mu * mu) * sin_val
                    + mu * (alpha_0 * beta_L - beta_0 * alpha_L) * cos_val)

        # Adjust index for pure Neumann (skip zero eigenvalue)
        search_index = index - 1 if is_pure_neumann else index

        # Define initial bracket
        pi_L = math.pi / L
        left = search_index * pi_L + 0.01 / L
        right = (search_index + 1) * pi_L - 0.01 / L

        # Try to bracket with expansion first
        try:
            left, right = RobinRootFinder.find_bracket_with_expansion(
                D, left, right
            )
        except RuntimeError:
            # Fall back to scanning
            left, right = RobinRootFinder.find_bracket_by_scanning(
                D, search_index * pi_L, (search_index + 1) * pi_L
            )

        # Find root by bisection
        return RobinRootFinder.bisect(D, left, right, tol=tol, maxit=maxit)

    @staticmethod
    def compute_coefficients_from_left_bc(
        mu: float,
        alpha_0: float,
        beta_0: float,
        alpha_L: float,
        beta_L: float,
        L: float
    ) -> Tuple[float, float]:
        """
        Compute coefficients (A, B) for Robin eigenfunction.

        The eigenfunction has form A cos(μx) + B sin(μx).
        From left BC: α₀ A + β₀ μ B = 0

        Args:
            mu: Eigenvalue
            alpha_0, beta_0: Left boundary coefficients
            alpha_L, beta_L: Right boundary coefficients
            L: Domain length

        Returns:
            Tuple (A, B) for the eigenfunction
        """
        if abs(alpha_0) >= abs(beta_0 * mu):
            # Normalize with A = 1
            A = 1.0
            if abs(alpha_0) > 1e-14:
                B = -alpha_0 / (beta_0 * mu) if abs(beta_0 * mu) > 1e-14 else 0.0
            else:
                B = 0.0
        else:
            # Normalize with B = 1
            A = -(beta_0 * mu) / alpha_0 if abs(alpha_0) > 1e-14 else 0.0
            B = 1.0

        return A, B
