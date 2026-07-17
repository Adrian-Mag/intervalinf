"""
Radial Laplacian eigenfunction providers.

These providers generate eigenfunctions for the radial Laplacian operator:
    L = -d²/dr² - (2/r)d/dr

which is self-adjoint with respect to the weighted inner product:
    ⟨f,g⟩ = ∫ f(r)g(r) r² dr

The eigenfunctions depend on the domain and boundary conditions:
- Domain (0, R) with regularity at r=0: Dirichlet or Neumann at R
- Domain (a, b) with 0 < a < b: DD, DN, ND, or NN boundary conditions
"""

import dataclasses
from typing import Union
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from .base import IndexedFunctionProvider
from ..utils.robin_utils import RobinRootFinder
from ..core.functions import Function


@dataclasses.dataclass(frozen=True)
class _GeneralRadialMode:
    """One normalized general-ell radial eigenmode."""

    k: float
    coeffs: tuple[float, float] | None
    norm: float


def _bc_pair(bc_type: str) -> tuple[str, str]:
    """Return endpoint BC labels for shell radial problems."""
    if bc_type == "dirichlet":
        return "D", "D"
    if bc_type == "neumann":
        return "N", "N"
    if bc_type == "mixed_dirichlet_neumann":
        return "D", "N"
    if bc_type == "mixed_neumann_dirichlet":
        return "N", "D"
    raise ValueError(f"Unsupported radial boundary condition {bc_type!r}")


def _j(ell: int, x):
    return spherical_jn(ell, x)


def _dj(ell: int, x):
    return spherical_jn(ell, x, derivative=True)


def _y(ell: int, x):
    return spherical_yn(ell, x)


def _dy(ell: int, x):
    return spherical_yn(ell, x, derivative=True)


class GeneralRadialLaplacianModeSolver:
    r"""Root and normalization solver for radial Laplacian modes with ``ell > 0``.

    The modes solve

    .. math::

        -f'' - 2r^{-1}f' + \ell(\ell+1)r^{-2}f = k^2 f

    and are normalized in ``L^2(r^2 dr)``. Domains containing zero use the
    regular solution ``j_ell(k r)``. Shell domains use
    ``A j_ell(k r) + B y_ell(k r)``.
    """

    def __init__(self, function_domain, boundary_conditions, ell: int):
        self.function_domain = function_domain
        self.boundary_conditions = boundary_conditions
        self.ell = int(ell)
        if self.ell <= 0:
            raise ValueError("GeneralRadialLaplacianModeSolver requires ell > 0")
        self.a = float(function_domain.a)
        self.b = float(function_domain.b)
        self.length = self.b - self.a
        if self.length <= 0.0:
            raise ValueError("Radial domain length must be positive")
        self._modes: list[_GeneralRadialMode] = []

        if np.isclose(self.a, 0.0, atol=1e-10):
            if boundary_conditions.type not in ("dirichlet", "neumann"):
                raise ValueError(
                    "Domains containing r=0 support only regularity plus "
                    f"outer dirichlet/neumann BCs, got {boundary_conditions.type!r}."
                )
        else:
            _bc_pair(boundary_conditions.type)

    def eigenvalue(self, index: int) -> float:
        """Return the unscaled eigenvalue ``k_n^2``."""
        return self.mode(index).k ** 2

    def mode(self, index: int) -> _GeneralRadialMode:
        """Return cached mode metadata for ``index``."""
        if index < 0:
            raise ValueError("Mode index must be non-negative")
        self._ensure_modes(index + 1)
        return self._modes[index]

    def values(self, index: int, r) -> np.ndarray:
        """Evaluate normalized mode ``index`` at radii ``r``."""
        mode = self.mode(index)
        r_arr = np.asarray(r, dtype=float)
        raw = self._raw_values(mode.k, mode.coeffs, r_arr)
        return mode.norm * raw

    def _ensure_modes(self, count: int) -> None:
        while len(self._modes) < count:
            root = self._find_root(len(self._modes))
            coeffs = None if np.isclose(self.a, 0.0, atol=1e-10) else self._left_coeffs(root)
            norm = self._normalization(root, coeffs, len(self._modes))
            self._modes.append(_GeneralRadialMode(root, coeffs, norm))

    def _characteristic(self, k: float) -> float:
        if k <= 0.0:
            return np.nan
        ell = self.ell
        if np.isclose(self.a, 0.0, atol=1e-10):
            x = k * self.b
            if self.boundary_conditions.type == "dirichlet":
                return float(_j(ell, x))
            if self.boundary_conditions.type == "neumann":
                return float(k * _dj(ell, x))
            raise AssertionError("Unsupported regular radial boundary condition")

        left_bc, right_bc = _bc_pair(self.boundary_conditions.type)
        coeffs = self._left_coeffs(k, left_bc=left_bc)
        return float(self._bc_row(k, self.b, right_bc) @ np.asarray(coeffs))

    def _characteristic_batch(self, ks: np.ndarray) -> np.ndarray:
        """Vectorized characteristic function over an array of k values.

        Replaces the Python loop in ``_scan_roots`` with a handful of
        scipy vectorized Bessel calls.  Returns NaN for non-positive k.
        """
        ell = self.ell
        out = np.full(len(ks), np.nan)
        valid = ks > 0.0
        if not np.any(valid):
            return out
        k = ks[valid]

        if np.isclose(self.a, 0.0, atol=1e-10):
            xb = k * self.b
            bc = self.boundary_conditions.type
            if bc == "dirichlet":
                out[valid] = spherical_jn(ell, xb)
            else:  # neumann
                out[valid] = k * spherical_jn(ell, xb, derivative=True)
            return out

        # Shell domain [a, b]: left-BC null vector × right-BC row.
        left_bc, right_bc = _bc_pair(self.boundary_conditions.type)
        xa = k * self.a
        xb = k * self.b

        # Left BC: row = _bc_row(k, a, left_bc), coeffs = (row[1], -row[0])
        if left_bc == "D":
            A = spherical_yn(ell, xa)
            B = -spherical_jn(ell, xa)
        else:  # "N"
            A = k * spherical_yn(ell, xa, derivative=True)
            B = -k * spherical_jn(ell, xa, derivative=True)

        # Right BC: row = _bc_row(k, b, right_bc), characteristic = row @ coeffs
        if right_bc == "D":
            out[valid] = spherical_jn(ell, xb) * A + spherical_yn(ell, xb) * B
        else:  # "N"
            out[valid] = (
                k * spherical_jn(ell, xb, derivative=True) * A
                + k * spherical_yn(ell, xb, derivative=True) * B
            )
        return out

    def _left_coeffs(self, k: float, left_bc: str | None = None) -> tuple[float, float]:
        if left_bc is None:
            left_bc, _ = _bc_pair(self.boundary_conditions.type)
        row = self._bc_row(k, self.a, left_bc)
        # Null vector for the one-row left boundary constraint.
        return (float(row[1]), float(-row[0]))

    def _bc_row(self, k: float, r: float, bc: str) -> np.ndarray:
        x = k * r
        if bc == "D":
            return np.array([_j(self.ell, x), _y(self.ell, x)], dtype=float)
        if bc == "N":
            return k * np.array([_dj(self.ell, x), _dy(self.ell, x)], dtype=float)
        raise ValueError(f"Unknown endpoint BC label {bc!r}")

    def _find_root(self, index: int) -> float:
        roots_needed = index + 1
        roots: list[float] = [m.k for m in self._modes]
        scale = self.b if np.isclose(self.a, 0.0, atol=1e-10) else self.length
        q_min = max(1e-4, roots[-1] * scale + 1e-6 if roots else 1e-4)
        q_max = max((roots_needed + self.ell + 8) * np.pi, q_min + 8 * np.pi)

        while len(roots) < roots_needed:
            roots.extend(self._scan_roots(q_min, q_max, scale, roots))
            if len(roots) >= roots_needed:
                break
            q_min = q_max
            q_max *= 1.6
            if q_max / scale > 1e8:
                raise RuntimeError(
                    f"Failed to find {roots_needed} roots for ell={self.ell}, "
                    f"BC={self.boundary_conditions.type!r} on "
                    f"({self.a}, {self.b})."
                )
        return roots[index]

    def _scan_roots(
        self,
        q_min: float,
        q_max: float,
        scale: float,
        existing_roots: list[float],
    ) -> list[float]:
        n_samples = max(512, int(np.ceil((q_max - q_min) / np.pi * 96)))
        qs = np.linspace(q_min, q_max, n_samples)
        ks = qs / scale
        vals = self._characteristic_batch(ks)
        roots: list[float] = []
        all_roots = list(existing_roots)

        for i in range(len(ks) - 1):
            v0 = vals[i]
            v1 = vals[i + 1]
            if not (np.isfinite(v0) and np.isfinite(v1)):
                continue
            k0 = ks[i]
            k1 = ks[i + 1]
            if v0 * v1 > 0.0:
                continue
            if abs(v0) < 1e-12 and abs(v1) < 1e-12:
                continue
            try:
                candidate = brentq(
                    self._characteristic, k0, k1,
                    xtol=1e-13, rtol=1e-12, maxiter=100,
                )
            except ValueError:
                continue

            if candidate <= 0.0:
                continue
            if all(abs(candidate - r) > 1e-7 * max(1.0, candidate) for r in all_roots):
                roots.append(float(candidate))
                all_roots.append(float(candidate))

        roots.sort()
        return roots

    def _raw_values(
        self,
        k: float,
        coeffs: tuple[float, float] | None,
        r: np.ndarray,
    ) -> np.ndarray:
        x = k * r
        if coeffs is None:
            return np.asarray(_j(self.ell, x), dtype=float)
        A, B = coeffs
        return np.asarray(A * _j(self.ell, x) + B * _y(self.ell, x), dtype=float)

    def _normalization(
        self,
        k: float,
        coeffs: tuple[float, float] | None,
        mode_index: int,
    ) -> float:
        n_points = max(4096, min(40000, 160 * (mode_index + self.ell + 2)))
        r = np.linspace(self.a, self.b, n_points)
        vals = self._raw_values(k, coeffs, r)
        integral = np.trapezoid(vals * vals * r * r, r)
        if not np.isfinite(integral) or integral <= 0.0:
            raise FloatingPointError(
                f"Invalid normalization integral for ell={self.ell}, "
                f"mode={mode_index}, k={k}."
            )
        return float(1.0 / np.sqrt(integral))


class GeneralRadialLaplacianProvider(IndexedFunctionProvider):
    """Eigenfunction provider for radial Laplacian modes with ``ell > 0``."""

    def __init__(self, space, boundary_conditions, ell: int, mode_solver=None):
        super().__init__(space)
        self._cache = {}
        self._mode_solver = mode_solver or GeneralRadialLaplacianModeSolver(
            space.function_domain, boundary_conditions, ell
        )
        self._ell = int(ell)

    @property
    def mode_solver(self) -> GeneralRadialLaplacianModeSolver:
        return self._mode_solver

    def get_function_by_index(self, index: int) -> Function:
        """Get the normalized general-ell radial eigenfunction."""
        if index not in self._cache:
            self._mode_solver.mode(index)

            def eigenfunction(r, _index=index, _solver=self._mode_solver):
                r_arr = np.asarray(r, dtype=float)
                scalar_input = r_arr.ndim == 0
                values = _solver.values(_index, np.atleast_1d(r_arr))
                return values.item() if scalar_input else values

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=f"y_{index}(r) [radial ell={self._ell}]",
            )
            self._cache[index] = func
        return self._cache[index]


class RadialLaplacianDirichletProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (0, R) with Dirichlet BC.

    Domain: (0, R) with regularity at r=0 and Dirichlet at r=R.
    Eigenfunctions: y_n(r) = φ_n(r)/r where φ_n(r) = √(2/R) sin(nπr/R)
    Eigenvalues: λ_n = (nπ/R)² for n=1,2,3,...
    """

    def __init__(self, space):
        """Initialize provider for (0,R) Dirichlet case."""
        super().__init__(space)
        self._cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not np.isclose(a, 0.0, atol=1e-10):
            raise ValueError(
                f"RadialLaplacianDirichletProvider requires domain starting at 0, "
                f"got a={a}"
            )

    def get_function_by_index(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Parameters
        ----------
        index : int
            Function index (0-based, maps to n=1,2,3,...)

        Returns
        -------
        Function
            Radial eigenfunction y_n(r) = φ_n(r)/r
        """
        if index not in self._cache:
            R = self.space.function_domain.b
            n = index + 1  # n starts from 1

            # Normalization constant for φ_n(r) = c_n sin(nπr/R)
            c_n = np.sqrt(2.0 / R)

            def eigenfunction(r, _n=n, _R=R, _c_n=c_n):
                """y_n(r) = c_n sin(nπr/R) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = np.zeros_like(r_arr, dtype=float)
                nonzero = r_arr > 1e-14

                # For r > 0: y_n(r) = c_n sin(nπr/R) / r
                result[nonzero] = (
                    _c_n * np.sin(_n * np.pi * r_arr[nonzero] / _R)
                    / r_arr[nonzero]
                )

                return result.item() if scalar_input else result

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=f"y_{n}(r) [radial Dirichlet]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianNeumannProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (0, R) with Neumann BC.

    Domain: (0, R) with regularity at r=0 and Neumann at r=R.

    Zero mode (index=0):
        y_0(r) = √(3/R³) (constant)
        λ_0 = 0

    Nonzero modes (index≥1):
        y_n(r) = c_n sin(k_n r) / r
        where k_n satisfies tan(k_n R) = k_n R
        Normalization: c_n = [R/2 - sin(2k_n R)/(4k_n)]^(-1/2)
        λ_n = k_n²
    """

    def __init__(self, space):
        """Initialize provider for (0,R) Neumann case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not np.isclose(a, 0.0, atol=1e-10):
            raise ValueError(
                f"RadialLaplacianNeumannProvider requires domain starting at 0, "
                f"got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            R = self.space.function_domain.b

            if index == 0:
                eigenval = 0.0
            else:
                # Solve tan(kR) = kR for the index-th root
                F = lambda k: k * R
                k_root = RobinRootFinder.solve_tan_equation(F, R, index - 1)
                eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Parameters
        ----------
        index : int
            Function index (0-based), index=0 gives zero mode,
            index≥1 gives nonzero modes

        Returns
        -------
        Function
            Radial eigenfunction
        """
        if index not in self._cache:
            R = self.space.function_domain.b

            if index == 0:
                # Zero mode: y_0(r) = constant
                c_0 = np.sqrt(3.0 / (R ** 3))

                def eigenfunction(r, _c_0=c_0):
                    """Zero mode: constant function"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    result = np.full_like(np.atleast_1d(r_arr), _c_0, dtype=float)
                    return result.item() if scalar_input else result

                name = f"y_0(r) [radial Neumann zero mode]"
            else:
                # Nonzero mode: solve for k_n
                F = lambda k: k * R
                k_n = RobinRootFinder.solve_tan_equation(F, R, index - 1)

                # Normalization integral: I_n = R/2 - sin(2k_n R)/(4k_n)
                I_n = R / 2.0 - np.sin(2 * k_n * R) / (4 * k_n)
                c_n = 1.0 / np.sqrt(I_n)

                def eigenfunction(r, _k_n=k_n, _c_n=c_n):
                    """y_n(r) = c_n sin(k_n r) / r"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    r_arr = np.atleast_1d(r_arr)

                    result = np.zeros_like(r_arr, dtype=float)
                    nonzero = r_arr > 1e-14

                    result[nonzero] = (
                        _c_n * np.sin(_k_n * r_arr[nonzero]) / r_arr[nonzero]
                    )

                    return result.item() if scalar_input else result

                name = f"y_{index}(r) [radial Neumann]"

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=name
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianDDProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Dirichlet-Dirichlet BC.

    Domain: (a, b) with 0 < a < b, Dirichlet at both endpoints.
    Eigenfunctions: y_n(r) = √(2/L) sin(nπ(r-a)/L) / r
    Eigenvalues: λ_n = (nπ/L)² for n=1,2,3,...
    where L = b - a
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Dirichlet-Dirichlet case."""
        super().__init__(space)
        self._cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianDDProvider requires domain with a > 0, got a={a}"
            )

    def get_function_by_index(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Parameters
        ----------
        index : int
            Function index (0-based, maps to n=1,2,3,...)

        Returns
        -------
        Function
            Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a
            n = index + 1  # n starts from 1

            c_n = np.sqrt(2.0 / L)

            def eigenfunction(r, _a=a, _L=L, _n=n, _c_n=c_n):
                """y_n(r) = √(2/L) sin(nπ(r-a)/L) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = _c_n * np.sin(_n * np.pi * (r_arr - _a) / _L) / r_arr

                return result.item() if scalar_input else result

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=f"y_{n}(r) [radial DD]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianDNProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Dirichlet-Neumann BC.

    Domain: (a, b) with 0 < a < b, Dirichlet at a, Neumann at b.
    Eigenfunctions: y_n(r) = c_n sin(k_n(r-a)) / r
    where k_n satisfies tan(k_n L) = k_n b
    Normalization: c_n = [L/2 - sin(2k_n L)/(4k_n)]^(-1/2)
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Dirichlet-Neumann case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianDNProvider requires domain with a > 0, got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = kb
            F = lambda k: k * b
            k_root = RobinRootFinder.solve_tan_equation(F, L, index)
            eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Parameters
        ----------
        index : int
            Function index (0-based)

        Returns
        -------
        Function
            Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = kb
            F = lambda k: k * b
            k_n = RobinRootFinder.solve_tan_equation(F, L, index)

            # Normalization: I_n = L/2 - sin(2k_n L)/(4k_n)
            I_n = L / 2.0 - np.sin(2 * k_n * L) / (4 * k_n)
            c_n = 1.0 / np.sqrt(I_n)

            def eigenfunction(r, _a=a, _k_n=k_n, _c_n=c_n):
                """y_n(r) = c_n sin(k_n(r-a)) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = _c_n * np.sin(_k_n * (r_arr - _a)) / r_arr

                return result.item() if scalar_input else result

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=f"y_{index}(r) [radial DN]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianNDProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Neumann-Dirichlet BC.

    Domain: (a, b) with 0 < a < b, Neumann at a, Dirichlet at b.
    Eigenfunctions: y_n(r) = c_n sin(k_n(b-r)) / r
    where k_n satisfies tan(k_n L) = -a k_n
    Normalization: c_n = [L/2 - sin(2k_n L)/(4k_n)]^(-1/2)
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Neumann-Dirichlet case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianNDProvider requires domain with a > 0, got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = -ak
            F = lambda k: -a * k
            k_root = RobinRootFinder.solve_tan_equation(F, L, index)
            eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Parameters
        ----------
        index : int
            Function index (0-based)

        Returns
        -------
        Function
            Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = -ak
            F = lambda k: -a * k
            k_n = RobinRootFinder.solve_tan_equation(F, L, index)

            # Normalization: I_n = L/2 - sin(2k_n L)/(4k_n)
            I_n = L / 2.0 - np.sin(2 * k_n * L) / (4 * k_n)
            c_n = 1.0 / np.sqrt(I_n)

            def eigenfunction(r, _b=b, _k_n=k_n, _c_n=c_n):
                """y_n(r) = c_n sin(k_n(b-r)) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = _c_n * np.sin(_k_n * (_b - r_arr)) / r_arr

                return result.item() if scalar_input else result

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=f"y_{index}(r) [radial ND]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianNNProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Neumann-Neumann BC.

    Domain: (a, b) with 0 < a < b, Neumann at both endpoints.

    Zero mode (index=0):
        y_0(r) = √(3/(b³-a³)) (constant)
        λ_0 = 0

    Nonzero modes (index≥1):
        y_n(r) = c_n [sin(k_n(r-a)) + ak_n cos(k_n(r-a))] / r
        where k_n satisfies tan(k_n L) = (1/b - 1/a) / (k + 1/(abk))
        Normalization from integral calculation
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Neumann-Neumann case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianNNProvider requires domain with a > 0, got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            if index == 0:
                eigenval = 0.0
            else:
                a = self.space.function_domain.a
                b = self.space.function_domain.b
                L = b - a

                # Solve tan(kL) = (1/b - 1/a) / (k + 1/(abk))
                numerator = 1.0/b - 1.0/a
                F = lambda k: numerator / (k + 1.0/(a * b * k))
                k_root = RobinRootFinder.solve_tan_equation(F, L, index - 1)
                eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Parameters
        ----------
        index : int
            Function index (0-based), index=0 gives zero mode,
            index≥1 gives nonzero modes

        Returns
        -------
        Function
            Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            if index == 0:
                # Zero mode: y_0(r) = constant
                c_0 = np.sqrt(3.0 / (b**3 - a**3))

                def eigenfunction(r, _c_0=c_0):
                    """Zero mode: constant function"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    result = np.full_like(np.atleast_1d(r_arr), _c_0, dtype=float)
                    return result.item() if scalar_input else result

                name = f"y_0(r) [radial NN zero mode]"
            else:
                # Nonzero mode: solve for k_n
                numerator = 1.0/b - 1.0/a
                F = lambda k: numerator / (k + 1.0/(a * b * k))
                k_n = RobinRootFinder.solve_tan_equation(F, L, index - 1)

                # Normalization: I_n = L/2*(1+(ak)²) + sin(2kL)/(4k)*((ak)²-1)
                #                      + ak*(1-cos(2kL))/(2k)
                ak = a * k_n
                I_n = (
                    L / 2.0 * (1 + ak**2)
                    + np.sin(2 * k_n * L) / (4 * k_n) * (ak**2 - 1)
                    + ak * (1 - np.cos(2 * k_n * L)) / (2 * k_n)
                )
                c_n = 1.0 / np.sqrt(I_n)

                def eigenfunction(r, _a=a, _k_n=k_n, _c_n=c_n):
                    """y_n(r) = c_n [sin(k_n(r-a)) + ak_n cos(k_n(r-a))] / r"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    r_arr = np.atleast_1d(r_arr)

                    ak_val = _a * _k_n
                    u_k = (
                        np.sin(_k_n * (r_arr - _a))
                        + ak_val * np.cos(_k_n * (r_arr - _a))
                    )
                    result = _c_n * u_k / r_arr

                    return result.item() if scalar_input else result

                name = f"y_{index}(r) [radial NN]"

            func = Function(
                self.space.function_domain,
                evaluate_callable=eigenfunction,
                name=name
            )
            self._cache[index] = func

        return self._cache[index]
