"""
Kernel utilities for normal-mode splitting coefficient sensitivity kernels.

This module provides classes for loading, managing, and providing normal-mode
sensitivity kernels from PREM-layer ``.kern`` files, and for associating them
with observed splitting coefficients from ``.new`` data files.

Coordinate convention
---------------------
Volume kernels (vp, vs, rho ``.kern`` files) store values indexed by
**PREM layer number**.  Each PREM layer spans a radial interval
``[r_start, r_end]`` (km from Earth centre).  This module converts layer IDs
to midpoint radii for interpolation.  Topography kernels (``topo`` ``.kern``
files) store values at specific PREM layers corresponding to Earth
discontinuities – in the current dataset, only layer 110 (the CMB, radius
≈ 3480 km, depth ≈ 2891 km) appears.

Kernel file format
------------------
::

    kern_{mode_id}_{param}_PREM-layers-all_PREM-iso.kern

Three columns, one row per ``(s_degree, prem_layer)`` pair::

    s_degree  prem_layer  kernel_value

where ``s_degree`` is the even spherical-harmonic degree of the
splitting-function coefficient (0, 2, 4, …, 2l) and ``prem_layer`` runs
from 1 to 222.

Observed data file format
-------------------------
::

    data_{dataset}/{mode_id}.new

First line: ``smax`` (maximum s degree in this file).
Subsequent lines::

    s  t  cst  cst_err

Key public API
--------------
EARTH_RADIUS_KM : float
    Earth's mean radius (6371.0 km).
DepthCoordinateSystem : class
    Static helpers for depth ↔ radius conversion.
TopoKernel : class
    Topography (boundary) sensitivity kernel stored as discrete depth/value
    pairs.  Same interface as in ``kernel_utils.py``.
NormalModeKernelCatalog : class
    Loads and caches kernel data from PREM-layer ``.kern`` files.
    Backward-compatible superset of ``SensitivityKernelCatalog``.
NormalModeDataRegistry : class
    Loads observed splitting coefficients from ``.new`` files and provides
    flat ``data_vector`` / ``error_vector`` arrays aligned with an ordered
    ``observations`` list of ``(mode_id, s, t)`` tuples.
NormalModeSplittingKernelProvider : IndexedFunctionProvider
    Provides one sensitivity ``Function`` per observation in a
    ``NormalModeDataRegistry``.  When ``registry=None`` falls back to one
    function per mode (backward-compatible mode).
    Inherits ``.restrict()`` from ``IndexedFunctionProvider``.
SensitivityKernelCatalog : alias for NormalModeKernelCatalog
SensitivityKernelProvider : alias for NormalModeSplittingKernelProvider

Backward compatibility
----------------------
``SensitivityKernelCatalog`` and ``SensitivityKernelProvider`` are aliases
for the new classes.  Code written against the old ``kernel_utils.py`` API
continues to work, with the following notes:

- ``SensitivityKernelCatalog.__init__(data_dir)`` is unchanged.
- ``SensitivityKernelProvider.__init__(space, catalog, interpolation_method,
  include_discontinuities, kernel_type)`` retains the same positional
  signature; the new ``registry`` parameter is appended as a keyword-only
  argument.  Default ``kernel_type`` is now ``"vs"`` (was ``"vp"``);
  pass ``kernel_type="vp"`` to restore the old default.
- Per-mode kernel accessors ``get_vp_kernel``, ``get_vs_kernel``,
  ``get_rho_kernel``, ``get_topo_kernel`` are retained with an extra optional
  ``s`` parameter (defaults match the old single-kernel behaviour).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.interpolate

from intervalinf import Function, IntervalDomain
from intervalinf.providers import IndexedFunctionProvider

# ---------------------------------------------------------------------------
# Earth constant
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6371.0

# PREM layer numbers for well-known boundaries.
PREM_LAYER_CMB: int = 110   # Core-Mantle Boundary (r ≈ 3480 km)
PREM_LAYER_ICB: int = 57    # Inner-Core Boundary (r ≈ 1220 km)

# ---------------------------------------------------------------------------
# Coordinate helpers (backward compatible)
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
# Topography kernel (backward compatible)
# ---------------------------------------------------------------------------


class TopoKernel:
    """
    Topography (boundary) sensitivity kernel stored as discrete depth/value
    pairs.

    Backward compatible with ``kernel_utils.TopoKernel``.

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
            Maximum allowed distance (km).  Returns ``None`` if no entry is
            within tolerance.

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
# PREM layer helpers (internal)
# ---------------------------------------------------------------------------


def _parse_prem_layers(filepath: Path) -> Dict[int, Tuple[float, float]]:
    """
    Parse ``PREM_all-layers_radius`` → ``{layer_id: (r_start_km, r_end_km)}``.

    Lines with ``r_end == 0`` are stored as-is; callers should use
    ``_layer_midpoint`` which handles the degenerate last-layer case.
    """
    layers: Dict[int, Tuple[float, float]] = {}
    with open(filepath) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            lid = int(parts[0])
            r_start = float(parts[1])
            r_end = float(parts[2])
            layers[lid] = (r_start, r_end)
    return layers


def _layer_midpoint(layers: Dict[int, Tuple[float, float]], layer_id: int) -> float:
    """
    Return the midpoint radius (km) of a PREM layer.

    For zero-thickness layers (discontinuity markers, r_start == r_end) and
    for the degenerate last layer (r_end == 0), returns r_start.
    """
    r_start, r_end = layers[layer_id]
    if r_end <= r_start or r_end == 0.0:
        return r_start
    return 0.5 * (r_start + r_end)


def _layer_boundary_radius(
    layers: Dict[int, Tuple[float, float]], layer_id: int
) -> float:
    """Return r_start of *layer_id* (the boundary on the shallow side)."""
    return layers[layer_id][0]


# ---------------------------------------------------------------------------
# NormalModeKernelCatalog
# ---------------------------------------------------------------------------


class NormalModeKernelCatalog:
    """
    Catalog of normal-mode splitting sensitivity kernels stored as PREM-layer
    ``.kern`` files.

    File-name convention::

        kern_{mode_id}_{param}_PREM-layers-all_PREM-iso.kern

    where *param* is one of ``vp``, ``vs``, ``rho``, ``topo``.

    Each file has three columns::

        s_degree  prem_layer  kernel_value

    Mode IDs are detected from the ``kern_*_vs_*.kern`` filenames.  The PREM
    layer → radius mapping is read from ``PREM_all-layers_radius`` (or a
    user-supplied path).

    Parameters
    ----------
    kernel_dir : str or Path
        Directory containing the ``.kern`` files.
    prem_radius_file : str or Path, optional
        Explicit path to ``PREM_all-layers_radius``.  Defaults to
        ``kernel_dir / "PREM_all-layers_radius"``.

    Notes
    -----
    Backward compatible with ``SensitivityKernelCatalog`` from
    ``kernel_utils.py``.
    """

    def __init__(self, kernel_dir, prem_radius_file=None) -> None:
        self.kernel_dir = Path(kernel_dir)
        if not self.kernel_dir.is_dir():
            raise FileNotFoundError(
                f"Kernel directory not found: {self.kernel_dir}"
            )
        if prem_radius_file is None:
            prem_radius_file = self.kernel_dir / "PREM_all-layers_radius"
        self._prem_file = Path(prem_radius_file)
        self._prem_layers: Optional[Dict[int, Tuple[float, float]]] = None
        self._mode_ids: Optional[List[str]] = None
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # PREM layer access
    # ------------------------------------------------------------------

    def _load_prem(self) -> Dict[int, Tuple[float, float]]:
        if self._prem_layers is None:
            self._prem_layers = _parse_prem_layers(self._prem_file)
        return self._prem_layers

    # ------------------------------------------------------------------
    # Mode discovery
    # ------------------------------------------------------------------

    def list_modes(self) -> List[str]:
        """Return a sorted list of all available mode IDs."""
        if self._mode_ids is None:
            ids: List[str] = []
            for fp in sorted(self.kernel_dir.glob("kern_*_vs_*.kern")):
                m = re.match(r"kern_(.+)_vs_", fp.name)
                if m:
                    ids.append(m.group(1))
            self._mode_ids = ids
        return self._mode_ids

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _load_kern_file(self, mode_id: str, param: str) -> np.ndarray:
        """
        Load and cache a ``.kern`` file.

        Returns an array of shape ``(N, 3)`` with columns
        ``[s_degree, prem_layer, kernel_value]``.
        """
        key = (mode_id, param)
        if key not in self._cache:
            fp = (
                self.kernel_dir
                / f"kern_{mode_id}_{param}_PREM-layers-all_PREM-iso.kern"
            )
            if not fp.exists():
                raise FileNotFoundError(f"Kernel file not found: {fp}")
            rows: List[Tuple[int, int, float]] = []
            with open(fp) as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
            self._cache[key] = np.array(rows, dtype=float)
        return self._cache[key]

    # ------------------------------------------------------------------
    # s-degree queries
    # ------------------------------------------------------------------

    def available_s_degrees(self, mode_id: str, param: str = "vs") -> List[int]:
        """
        Return the sorted list of spherical-harmonic degrees present in the
        kernel file for *mode_id* and *param*.
        """
        data = self._load_kern_file(mode_id, param)
        return sorted({int(s) for s in data[:, 0]})

    # ------------------------------------------------------------------
    # Volume kernel accessors
    # ------------------------------------------------------------------

    def get_volume_data(
        self,
        mode_id: str,
        param: str,
        s_degree: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(radii_km, kernel_values)`` for the given *s_degree*.

        Radii are the midpoints of the PREM layers, in km from Earth centre.

        Parameters
        ----------
        mode_id : str
            Mode identifier, e.g. ``"00s04"``.
        param : str
            Physical parameter: ``"vp"``, ``"vs"``, or ``"rho"``.
        s_degree : int
            Spherical-harmonic degree to extract.

        Returns
        -------
        radii_km : np.ndarray, shape (N,)
        values : np.ndarray, shape (N,)
        """
        data = self._load_kern_file(mode_id, param)
        prem = self._load_prem()
        mask = data[:, 0] == s_degree
        rows = data[mask]
        layer_ids = rows[:, 1].astype(int)
        values = rows[:, 2]
        radii = np.array([_layer_midpoint(prem, lid) for lid in layer_ids])
        return radii, values

    def get_vp_data(
        self, mode_id: str, s_degree: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(radii_km, K_vp)`` for *s_degree* (default 0)."""
        return self.get_volume_data(mode_id, "vp", s_degree)

    def get_vs_data(
        self, mode_id: str, s_degree: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(radii_km, K_vs)`` for *s_degree*.

        If *s_degree* is ``None``, the second available degree is used (the
        first nonzero even degree, typically 2), falling back to 0 if the
        file contains only s=0.
        """
        if s_degree is None:
            available = self.available_s_degrees(mode_id, "vs")
            s_degree = available[1] if len(available) > 1 else available[0]
        return self.get_volume_data(mode_id, "vs", s_degree)

    def get_rho_data(
        self, mode_id: str, s_degree: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(radii_km, K_rho)`` for *s_degree* (default 0)."""
        return self.get_volume_data(mode_id, "rho", s_degree)

    # ------------------------------------------------------------------
    # Topography kernel accessors
    # ------------------------------------------------------------------

    def get_topo_data(
        self, mode_id: str, s_degree: int = 0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return ``(depths_km_from_surface, kernel_values)`` for *s_degree*,
        or ``None`` if the topo file does not exist.

        Backward compatible with ``SensitivityKernelCatalog.get_topo_data``.
        Depths use the ``r_start`` of each PREM layer (the boundary radius)
        converted to depth from surface.
        """
        try:
            data = self._load_kern_file(mode_id, "topo")
        except FileNotFoundError:
            return None
        prem = self._load_prem()
        mask = data[:, 0] == s_degree
        rows = data[mask]
        if len(rows) == 0:
            return None
        layer_ids = rows[:, 1].astype(int)
        values = rows[:, 2]
        # Use r_start of each layer so the depth matches the actual boundary.
        depths = np.array(
            [EARTH_RADIUS_KM - _layer_boundary_radius(prem, lid) for lid in layer_ids]
        )
        return depths, values

    def get_topo_kernel(
        self, mode_id: str, s_degree: int = 0
    ) -> Optional[TopoKernel]:
        """
        Return a ``TopoKernel`` for *mode_id* and *s_degree*, or ``None``
        if no topo file exists.
        """
        result = self.get_topo_data(mode_id, s_degree)
        if result is None:
            return None
        depths, values = result
        return TopoKernel(depths, values)

    def get_topo_layer_value(
        self, mode_id: str, s_degree: int, prem_layer: int
    ) -> float:
        """
        Return the scalar topographic kernel value for *mode_id*, *s_degree*,
        and *prem_layer*.  Returns 0.0 if not found.
        """
        try:
            data = self._load_kern_file(mode_id, "topo")
        except FileNotFoundError:
            return 0.0
        mask = (data[:, 0] == s_degree) & (data[:, 1] == prem_layer)
        rows = data[mask]
        if len(rows) == 0:
            return 0.0
        return float(rows[0, 2])


# ---------------------------------------------------------------------------
# NormalModeDataRegistry
# ---------------------------------------------------------------------------


@dataclass
class _Observation:
    mode_id: str
    s: int
    t: int
    cst: float
    cst_err: float


class NormalModeDataRegistry:
    """
    Registry of observed normal-mode splitting coefficients.

    Reads all ``.new`` files for a chosen dataset (``"arwen-paula"`` or
    ``"SP12RTS"``) and exposes a flat ordered list of ``(mode_id, s, t)``
    observations with corresponding data and error arrays.

    Parameters
    ----------
    data_dir : str or Path
        Path to the ``normal-mode-data`` directory that contains the
        ``data_list_*`` files and ``data_*`` subdirectories.
    dataset : str, optional
        Either ``"arwen-paula"`` (default) or ``"SP12RTS"``.
    mode_filter : list of str, optional
        If provided, only include modes whose IDs appear in this list.
        Useful for selecting a sub-catalogue (e.g., modes that also have
        kernels).
    """

    _SUBDIRS: Dict[str, str] = {
        "arwen-paula": "data_arwen-paula",
        "SP12RTS": "data_SP12RTS",
    }
    _LISTS: Dict[str, str] = {
        "arwen-paula": "data_list_arwen-paula",
        "SP12RTS": "data_list_SP12RTS",
    }

    def __init__(
        self,
        data_dir,
        dataset: str = "arwen-paula",
        mode_filter: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if dataset not in self._SUBDIRS:
            raise ValueError(
                f"Unknown dataset {dataset!r}. "
                f"Choose from {sorted(self._SUBDIRS)}"
            )
        self.dataset = dataset
        self.mode_filter = mode_filter
        self._all_modes: Optional[List[str]] = None
        self._obs_list: Optional[List[_Observation]] = None

    # ------------------------------------------------------------------
    # Mode list
    # ------------------------------------------------------------------

    def list_modes(self) -> List[str]:
        """
        Return the ordered list of mode IDs in this dataset (after
        applying any ``mode_filter``).
        """
        if self._all_modes is None:
            list_file = self.data_dir / self._LISTS[self.dataset]
            with open(list_file) as fh:
                all_modes = [line.strip() for line in fh if line.strip()]
            if self.mode_filter is not None:
                filter_set = set(self.mode_filter)
                all_modes = [m for m in all_modes if m in filter_set]
            self._all_modes = all_modes
        return self._all_modes

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        if self._obs_list is not None:
            return
        subdir = self.data_dir / self._SUBDIRS[self.dataset]
        obs: List[_Observation] = []
        for mode_id in self.list_modes():
            fp = subdir / f"{mode_id}.new"
            if not fp.exists():
                continue
            with open(fp) as fh:
                lines = fh.readlines()
            # First line: smax.  Remaining lines: s t cst cst_err.
            for line in lines[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                obs.append(
                    _Observation(
                        mode_id=mode_id,
                        s=int(parts[0]),
                        t=int(parts[1]),
                        cst=float(parts[2]),
                        cst_err=float(parts[3]),
                    )
                )
        self._obs_list = obs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def observations(self) -> List[Tuple[str, int, int]]:
        """
        Flat ordered list of ``(mode_id, s, t)`` tuples, one per
        splitting-coefficient observation.
        """
        self._load_all()
        return [(o.mode_id, o.s, o.t) for o in self._obs_list]  # type: ignore[union-attr]

    @property
    def data_vector(self) -> np.ndarray:
        """Observed splitting coefficients, shape ``(N,)``."""
        self._load_all()
        return np.array([o.cst for o in self._obs_list])  # type: ignore[union-attr]

    @property
    def error_vector(self) -> np.ndarray:
        """Reported errors ``σ``, shape ``(N,)``."""
        self._load_all()
        return np.array([o.cst_err for o in self._obs_list])  # type: ignore[union-attr]

    @property
    def covariance_diagonal(self) -> np.ndarray:
        """Diagonal of the data covariance matrix ``σ²``, shape ``(N,)``."""
        return self.error_vector ** 2

    def __len__(self) -> int:
        self._load_all()
        return len(self._obs_list)  # type: ignore[arg-type]

    def find_index(self, mode_id: str, s: int, t: int) -> int:
        """Return the flat index of the observation ``(mode_id, s, t)``."""
        self._load_all()
        key = (mode_id, s, t)
        for i, obs in enumerate(self._obs_list):  # type: ignore[union-attr]
            if (obs.mode_id, obs.s, obs.t) == key:
                return i
        raise KeyError(f"Observation {key!r} not found in registry")

    def filter_by_st(
        self,
        s: Optional[int] = None,
        t: Optional[int] = None,
    ) -> "NormalModeDataRegistry":
        """
        Return a new registry containing only observations that match the
        given spherical-harmonic degree *s* and/or azimuthal order *t*.

        Pass ``None`` for either parameter to leave it unconstrained (acts
        as a wildcard).

        Parameters
        ----------
        s : int or None
            If not ``None``, keep only observations with this s-degree.
        t : int or None
            If not ``None``, keep only observations with this t-order.

        Returns
        -------
        NormalModeDataRegistry
            A new registry whose ``observations``, ``data_vector``, and
            ``error_vector`` reflect the filtered subset.  The new registry
            shares the same underlying data directory and dataset; no files
            are re-read.

        Examples
        --------
        Select all observations for the s=2 splitting coefficients::

            reg_s2 = reg.filter_by_st(s=2)

        Select the single (s=2, t=0) subset (one obs per mode)::

            reg_s2t0 = reg.filter_by_st(s=2, t=0)
        """
        self._load_all()
        filtered = [
            o for o in self._obs_list  # type: ignore[union-attr]
            if (s is None or o.s == s) and (t is None or o.t == t)
        ]
        new_reg: "NormalModeDataRegistry" = NormalModeDataRegistry.__new__(
            NormalModeDataRegistry
        )
        new_reg.data_dir = self.data_dir
        new_reg.dataset = self.dataset
        new_reg.mode_filter = list({o.mode_id for o in filtered})
        new_reg._all_modes = [o.mode_id for o in filtered]
        new_reg._obs_list = filtered
        return new_reg


# ---------------------------------------------------------------------------
# NormalModeSplittingKernelProvider
# ---------------------------------------------------------------------------


class NormalModeSplittingKernelProvider(IndexedFunctionProvider):
    """
    ``IndexedFunctionProvider`` for normal-mode splitting sensitivity kernels.

    Each integer index corresponds to one observed splitting coefficient
    ``(mode_id, s, t)`` from a ``NormalModeDataRegistry``.  The returned
    ``Function`` is the volumetric kernel ``K_{mode, s}(r)`` for the chosen
    *kernel_type* (``"vp"``, ``"vs"``, or ``"rho"``).  Because the kernel
    does not depend on the azimuthal order *t*, observations sharing the same
    ``(mode_id, s)`` receive identical ``Function`` objects.

    When *registry* is ``None``, falls back to one kernel per mode using
    the first available *s* degree – this replicates the behaviour of the
    old ``SensitivityKernelProvider``.

    Inherits ``.restrict(sub_space)`` from ``IndexedFunctionProvider``.

    Parameters
    ----------
    space : Lebesgue
        Model space (radial domain, km from Earth centre).
    catalog : NormalModeKernelCatalog
        Kernel data source.
    interpolation_method : str, optional
        Kind passed to ``scipy.interpolate.interp1d`` (default ``"linear"``).
    include_discontinuities : bool, optional
        Reserved for future use; currently has no effect.
    kernel_type : str, optional
        ``"vp"``, ``"vs"``, or ``"rho"``.  Default ``"vs"``.
    registry : NormalModeDataRegistry, optional
        When provided, the provider is indexed by the registry's observations.
        When ``None``, indexed by ``catalog.list_modes()`` (backward-compat).
    """

    def __init__(
        self,
        space,
        catalog: NormalModeKernelCatalog,
        interpolation_method: str = "linear",
        include_discontinuities: bool = False,
        kernel_type: str = "vs",
        *,
        registry: Optional[NormalModeDataRegistry] = None,
    ) -> None:
        super().__init__(space)
        self.catalog = catalog
        self.interpolation_method = interpolation_method
        self.include_discontinuities = include_discontinuities
        self.kernel_type = kernel_type
        self.registry = registry

        if registry is not None:
            self._obs_keys: List[Tuple[str, int, int]] = registry.observations
        else:
            # Backward-compat: one entry per mode, s=first available degree.
            self._obs_keys = [
                (mid, catalog.available_s_degrees(mid, kernel_type)[0], 0)
                for mid in catalog.list_modes()
            ]

    # ------------------------------------------------------------------
    # Internal: build a Function from (radii, values) arrays
    # ------------------------------------------------------------------

    def _make_function(
        self,
        radii_km: np.ndarray,
        kernel_vals: np.ndarray,
        target_space=None,
    ) -> Function:
        """
        Interpolate kernel data and return a ``Function`` on *target_space*
        (defaults to ``self.space``).

        Data must already be in radius-from-centre coordinates (km).
        Duplicate radii are de-duplicated (last value kept after sorting).
        Points outside the data range evaluate to 0.
        """
        space = target_space if target_space is not None else self._space_or_domain

        radii = np.asarray(radii_km, dtype=float)
        vals = np.asarray(kernel_vals, dtype=float)

        sort_idx = np.argsort(radii)
        radii = radii[sort_idx]
        vals = vals[sort_idx]

        _, uniq_idx = np.unique(radii, return_index=True)
        radii = radii[uniq_idx]
        vals = vals[uniq_idx]

        interp = scipy.interpolate.interp1d(
            radii,
            vals,
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
        """
        Return the kernel ``Function`` for the *index*-th observation.

        The kernel is ``K_{mode_id, s}(r)`` for the *kernel_type*, where
        ``(mode_id, s, t) = self._obs_keys[index]``.
        """
        if index < 0 or index >= len(self._obs_keys):
            raise IndexError(
                f"Index {index} out of range [0, {len(self._obs_keys)})"
            )
        mode_id, s, _ = self._obs_keys[index]
        radii, vals = self.catalog.get_volume_data(mode_id, self.kernel_type, s)
        return self._make_function(radii, vals)

    def __len__(self) -> int:
        return len(self._obs_keys)

    # ------------------------------------------------------------------
    # Named kernel accessors (by mode ID and s degree)
    # ------------------------------------------------------------------

    def get_vp_kernel(
        self, mode_id: str, s: int = 0, space=None
    ) -> Function:
        """Return the vp sensitivity ``Function`` for *mode_id* at degree *s*."""
        radii, vals = self.catalog.get_vp_data(mode_id, s)
        return self._make_function(radii, vals, target_space=space)

    def get_vs_kernel(
        self, mode_id: str, s: Optional[int] = None, space=None
    ) -> Function:
        """
        Return the vs sensitivity ``Function`` for *mode_id* at degree *s*.

        If *s* is ``None``, uses the same default as
        ``NormalModeKernelCatalog.get_vs_data``.
        """
        radii, vals = self.catalog.get_vs_data(mode_id, s)
        return self._make_function(radii, vals, target_space=space)

    def get_rho_kernel(
        self, mode_id: str, s: int = 0, space=None
    ) -> Function:
        """Return the rho sensitivity ``Function`` for *mode_id* at degree *s*."""
        radii, vals = self.catalog.get_rho_data(mode_id, s)
        return self._make_function(radii, vals, target_space=space)

    def get_topo_kernel(
        self, mode_id: str, s: int = 0
    ) -> Optional[TopoKernel]:
        """
        Return the topography ``TopoKernel`` for *mode_id* at degree *s*,
        or ``None`` if no topo file exists.
        """
        return self.catalog.get_topo_kernel(mode_id, s)

    # ------------------------------------------------------------------
    # Topographic kernel matrix
    # ------------------------------------------------------------------

    def build_topo_matrix(
        self,
        prem_layer: int = PREM_LAYER_CMB,
        n_obs: Optional[int] = None,
    ) -> np.ndarray:
        """
        Build a vector of topographic kernel values at a single PREM boundary
        layer, aligned with the observation list.

        Each entry is the scalar sensitivity of observation ``(mode_id, s, t)``
        to topographic perturbation at *prem_layer*.

        Parameters
        ----------
        prem_layer : int, optional
            PREM layer number.  Default is ``PREM_LAYER_CMB`` (110, CMB).
        n_obs : int, optional
            If provided, only the first *n_obs* observations are used.

        Returns
        -------
        np.ndarray, shape ``(N_obs,)``
            Topographic sensitivity values.
        """
        keys = self._obs_keys[:n_obs] if n_obs is not None else self._obs_keys
        out: List[float] = []
        for mode_id, s, _ in keys:
            v = self.catalog.get_topo_layer_value(mode_id, s, prem_layer)
            out.append(v)
        return np.array(out)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

#: Alias for :class:`NormalModeKernelCatalog`.
SensitivityKernelCatalog = NormalModeKernelCatalog

#: Alias for :class:`NormalModeSplittingKernelProvider`.
SensitivityKernelProvider = NormalModeSplittingKernelProvider
