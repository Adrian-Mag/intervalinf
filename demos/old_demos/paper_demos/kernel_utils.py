"""
Kernel utilities for seismic sensitivity kernel loading and interpolation.

This module provides classes for loading, managing, and providing seismic
sensitivity kernels from .dat files.

Coordinate convention
---------------------
The .dat files store kernel values indexed by **depth from surface** (km):
  depth = 0  → surface
  depth = 6371 → Earth centre

The ``intervalinf`` Lebesgue-space domain uses **radius from centre** (km):
  radius = 0    → Earth centre
  radius = 6371 → surface

Conversion: ``radius = EARTH_RADIUS_KM - depth``

Key public API
--------------
EARTH_RADIUS_KM : float
    Earth's mean radius (6371.0 km).
DepthCoordinateSystem : class
    Static helpers for depth ↔ radius conversion.
SensitivityKernelCatalog : class
    Loads and caches kernel data from a directory of .dat files.
SensitivityKernelProvider : IndexedFunctionProvider
    Wraps a catalog so that a ``SOLAOperator`` can iterate over modes.
    Also exposes ``get_vp_kernel``, ``get_vs_kernel``, ``get_rho_kernel``,
    and ``get_topo_kernel`` for direct access by mode ID.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate
from pathlib import Path
from typing import Optional, List, Tuple

from intervalinf import Function, IntervalDomain
from intervalinf.providers import IndexedFunctionProvider

# ---------------------------------------------------------------------------
# Earth constant
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6371.0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

class DepthCoordinateSystem:
    """Conversion helpers between depth-from-surface and radius-from-centre."""

    earth_radius: float = EARTH_RADIUS_KM

    @staticmethod
    def depth_to_radius(depth_km: float) -> float:
        """Convert depth from surface (km) → radius from centre (km)."""
        return EARTH_RADIUS_KM - depth_km

    @staticmethod
    def radius_to_depth(radius_km: float) -> float:
        """Convert radius from centre (km) → depth from surface (km)."""
        return EARTH_RADIUS_KM - radius_km


# ---------------------------------------------------------------------------
# Topography kernel object
# ---------------------------------------------------------------------------

class TopoKernel:
    """
    Topography (boundary) sensitivity kernel stored as discrete depth/value
    pairs.

    File format: each row is ``depth_from_surface_km  kernel_value``.
    Rows marked with ``#`` (comment lines) are skipped.

    Parameters
    ----------
    depths_km : array-like
        Discontinuity depths from surface (km).
    values : array-like
        Corresponding topography kernel values.
    """

    def __init__(self, depths_km: np.ndarray, values: np.ndarray) -> None:
        self.depths = np.asarray(depths_km, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def get_value_at_depth(
        self,
        depth_km: float,
        tolerance: float = 100.0,
    ) -> Optional[float]:
        """
        Return the kernel value at the discontinuity closest to *depth_km*.

        Parameters
        ----------
        depth_km : float
            Target depth from surface (km).
        tolerance : float, optional
            Maximum allowed distance (km) to the closest stored depth.
            Returns ``None`` if no entry is within tolerance.

        Returns
        -------
        float or None
        """
        diffs = np.abs(self.depths - depth_km)
        i = int(np.argmin(diffs))
        if diffs[i] <= tolerance:
            return float(self.values[i])
        return None


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class SensitivityKernelCatalog:
    """
    Catalog of seismic sensitivity kernels stored as individual ``.dat``
    files in a directory.

    Expected file naming convention::

        vp-sens_{mode_id}_iso.dat   – P-velocity kernels
        vs-sens_{mode_id}_iso.dat   – S-velocity kernels
        rho-sens_{mode_id}_iso.dat  – Density kernels
        topo_kernels_{mode_id}_iso.dat – Topography (boundary) kernels

    All volume-kernel files (vp, vs, rho) are **two-column** with no header::

        depth_km   kernel_value
        0.000      0.00000E+00
        22.620     6.30000E-05
        ...

    Topography-kernel files have two ``#``-prefixed header lines followed by
    the same two-column format.

    Mode IDs are detected from the ``vp-sens_*_iso.dat`` filenames.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the ``.dat`` files.
    """

    def __init__(self, data_dir) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Kernel data directory not found: {self.data_dir}"
            )
        self._mode_ids: Optional[List[str]] = None
        self._cache: dict = {}  # (mode_id, kernel_type) → (depths, values)

    # ------------------------------------------------------------------
    # Mode discovery
    # ------------------------------------------------------------------

    def list_modes(self) -> List[str]:
        """Return a sorted list of all available mode IDs."""
        if self._mode_ids is None:
            files = sorted(self.data_dir.glob("vp-sens_*_iso.dat"))
            self._mode_ids = [
                f.stem.replace("vp-sens_", "").replace("_iso", "")
                for f in files
            ]
        return self._mode_ids

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_two_col(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a two-column (depth, value) file, skipping ``#`` lines."""
        depths: List[float] = []
        values: List[float] = []
        with open(filepath) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                depths.append(float(parts[0]))
                values.append(float(parts[1]))
        return np.array(depths), np.array(values)

    def _get_data(
        self,
        mode_id: str,
        kernel_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load (and cache) raw (depth_km, values) arrays."""
        key = (mode_id, kernel_type)
        if key not in self._cache:
            prefix_map = {
                "vp": "vp-sens",
                "vs": "vs-sens",
                "rho": "rho-sens",
                "topo": "topo_kernels",
            }
            prefix = prefix_map[kernel_type]
            fp = self.data_dir / f"{prefix}_{mode_id}_iso.dat"
            if not fp.exists():
                raise FileNotFoundError(f"Kernel file not found: {fp}")
            self._cache[key] = self._load_two_col(fp)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Named accessors
    # ------------------------------------------------------------------

    def get_vp_data(self, mode_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (depths_km, K_vp) for the given mode."""
        return self._get_data(mode_id, "vp")

    def get_vs_data(self, mode_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (depths_km, K_vs) for the given mode."""
        return self._get_data(mode_id, "vs")

    def get_rho_data(self, mode_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (depths_km, K_rho) for the given mode."""
        return self._get_data(mode_id, "rho")

    def get_topo_data(
        self, mode_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (depths_km, K_topo) for the given mode, or ``None`` if the
        file does not exist.
        """
        fp = self.data_dir / f"topo_kernels_{mode_id}_iso.dat"
        if not fp.exists():
            return None
        return self._load_two_col(fp)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class SensitivityKernelProvider(IndexedFunctionProvider):
    """
    ``IndexedFunctionProvider`` that supplies seismic sensitivity kernels
    as ``Function`` objects.

    Use this class directly with ``SOLAOperator``.  It can also return
    kernels by mode ID via the ``get_vp_kernel`` / ``get_vs_kernel`` /
    ``get_rho_kernel`` / ``get_topo_kernel`` helpers.

    Coordinate system
    ~~~~~~~~~~~~~~~~~
    The raw files use depth from surface (km); the provider automatically
    converts to radius from centre (km) so that the returned ``Function``
    objects live on the same grid as the ``Lebesgue`` model space.

    The returned functions are created with an ``evaluate_callable``, so
    they support ``.restrict(sub_space)`` out of the box.

    Parameters
    ----------
    space : Lebesgue
        The model space for which kernels are provided.
    catalog : SensitivityKernelCatalog
        The kernel catalog to draw data from.
    interpolation_method : str, optional
        Interpolation kind passed to ``scipy.interpolate.interp1d``
        (e.g. ``"linear"``, ``"cubic"``).  Default ``"linear"``.
    include_discontinuities : bool, optional
        *Reserved for future use* – discontinuity handling at ICB/CMB.
        Currently has no effect on the returned functions.
    kernel_type : str, optional
        Which kernel component to expose via ``get_function_by_index``:
        ``"vp"``, ``"vs"``, or ``"rho"``.  Default ``"vp"``.
    """

    def __init__(
        self,
        space,
        catalog: SensitivityKernelCatalog,
        interpolation_method: str = "linear",
        include_discontinuities: bool = False,
        kernel_type: str = "vp",
    ) -> None:
        super().__init__(space)
        self.catalog = catalog
        self.interpolation_method = interpolation_method
        self.include_discontinuities = include_discontinuities
        self.kernel_type = kernel_type
        self._mode_ids = catalog.list_modes()

    # ------------------------------------------------------------------
    # Internal: build a Function from raw depth/value arrays
    # ------------------------------------------------------------------

    def _make_function(
        self,
        depths_km: np.ndarray,
        kernel_vals: np.ndarray,
        target_space=None,
    ) -> Function:
        """
        Interpolate kernel data (depth coordinates) and return a
        ``Function`` on *target_space* (defaults to ``self.space``).
        """
        space = target_space if target_space is not None else self._space_or_domain

        # Convert depth → radius, then sort by ascending radius
        radii = EARTH_RADIUS_KM - depths_km
        sort_idx = np.argsort(radii)
        radii = radii[sort_idx]
        kernel_vals = kernel_vals[sort_idx]

        # Remove duplicate radii (keep last occurrence after sorting)
        _, unique_idx = np.unique(radii, return_index=True)
        radii = radii[unique_idx]
        kernel_vals = kernel_vals[unique_idx]

        # Build interpolator (extend beyond data range with 0)
        interp = scipy.interpolate.interp1d(
            radii,
            kernel_vals,
            kind=self.interpolation_method,
            bounds_error=False,
            fill_value=0.0,
        )

        def _eval(x: np.ndarray) -> np.ndarray:
            return interp(np.asarray(x, dtype=float))

        return Function(space, evaluate_callable=_eval)

    # ------------------------------------------------------------------
    # IndexedFunctionProvider interface
    # ------------------------------------------------------------------

    def get_function_by_index(self, index: int, **kwargs) -> Function:
        """Return the kernel ``Function`` for the *index*-th mode."""
        if index < 0 or index >= len(self._mode_ids):
            raise IndexError(
                f"Kernel index {index} out of range [0, {len(self._mode_ids)})"
            )
        mode_id = self._mode_ids[index]
        depths, vals = self.catalog._get_data(mode_id, self.kernel_type)
        return self._make_function(depths, vals)

    def __len__(self) -> int:
        return len(self._mode_ids)

    # ------------------------------------------------------------------
    # Named kernel accessors (by mode ID)
    # ------------------------------------------------------------------

    def get_vp_kernel(self, mode_id: str, space=None) -> Function:
        """Return the vp sensitivity ``Function`` for *mode_id*."""
        depths, vals = self.catalog.get_vp_data(mode_id)
        return self._make_function(depths, vals, target_space=space)

    def get_vs_kernel(self, mode_id: str, space=None) -> Function:
        """Return the vs sensitivity ``Function`` for *mode_id*."""
        depths, vals = self.catalog.get_vs_data(mode_id)
        return self._make_function(depths, vals, target_space=space)

    def get_rho_kernel(self, mode_id: str, space=None) -> Function:
        """Return the rho sensitivity ``Function`` for *mode_id*."""
        depths, vals = self.catalog.get_rho_data(mode_id)
        return self._make_function(depths, vals, target_space=space)

    def get_topo_kernel(self, mode_id: str) -> Optional[TopoKernel]:
        """
        Return the topography ``TopoKernel`` for *mode_id*, or ``None`` if
        the corresponding file does not exist.
        """
        data = self.catalog.get_topo_data(mode_id)
        if data is None:
            return None
        depths, values = data
        return TopoKernel(depths, values)
