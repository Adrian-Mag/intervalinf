"""
Configuration objects for numerical integration and parallelization.

This module provides dataclasses for managing integration and parallel
computation parameters across different subsystems.
"""

from dataclasses import dataclass, field
from typing import Literal
import copy as copy_module

# Methods that build a fixed-point mesh and are therefore amenable to
# automatic batching (Phase 4).  Changing this set is the only thing
# needed to opt a new method into future accelerated paths.
FIXED_GRID_METHODS: frozenset = frozenset({"simpson", "trapz"})

# Methods that delegate entirely to scipy.integrate.quad (adaptive).
# 'quad' is the legacy name; 'adaptive' is the canonical name. Both are
# supported wherever integration methods are accepted.
ADAPTIVE_METHODS: frozenset = frozenset({"adaptive", "quad"})


@dataclass
class IntegrationConfig:
    """
    Configuration for numerical integration parameters.

    Parameters
    ----------
    method : {'simpson', 'trapz', 'adaptive', 'quad'}
        Integration method.

        - ``'simpson'`` / ``'trapz'`` — fixed-grid quadrature rules that
          build a uniform mesh of *n_points* nodes.  These are
          **fixed-grid** methods that are candidates for automatic batched
          acceleration (Phase 4).
        - ``'adaptive'`` — delegates to ``scipy.integrate.quad`` for
          adaptive error-controlled integration.  Using this method means
          ``n_points`` is ignored.
        - ``'quad'`` — **alias** for ``'adaptive'`` kept for backward
          compatibility.  New code should prefer ``'adaptive'``.

    n_points : int
        Number of quadrature points.  Applies only to fixed-grid methods
        (``'simpson'`` and ``'trapz'``); ignored for adaptive methods.

    Examples
    --------
    >>> config = IntegrationConfig()  # defaults
    >>> config = IntegrationConfig(n_points=10000)
    >>> high_acc = IntegrationConfig.high_accuracy()
    >>> adapt = IntegrationConfig.adaptive_quad()
    """

    method: Literal["simpson", "trapz", "adaptive", "quad"] = "simpson"
    n_points: int = 1000

    @property
    def is_fixed_grid(self) -> bool:
        """True if this method builds a fixed-point mesh (batchable in Phase 4)."""
        return self.method in FIXED_GRID_METHODS

    @property
    def is_adaptive(self) -> bool:
        """True if this method uses adaptive (scipy.integrate.quad) integration."""
        return self.method in ADAPTIVE_METHODS

    def copy(self, **overrides) -> "IntegrationConfig":
        """Create a copy with optional parameter overrides."""
        new_config = copy_module.copy(self)
        for key, value in overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return new_config

    @classmethod
    def high_accuracy(cls) -> "IntegrationConfig":
        """Preset for high-accuracy integration."""
        return cls(method="simpson", n_points=10000)

    @classmethod
    def fast(cls) -> "IntegrationConfig":
        """Preset for fast, lower-accuracy integration."""
        return cls(method="trapz", n_points=500)

    @classmethod
    def adaptive_quad(cls) -> "IntegrationConfig":
        """Preset for adaptive (scipy.integrate.quad) integration."""
        return cls(method="adaptive")

    @classmethod
    def adaptive(cls, dim: int) -> "IntegrationConfig":
        """
        Create config with points scaled to basis dimension.

        For spectral methods, higher mode numbers require more integration
        points to maintain accuracy.

        Parameters
        ----------
        dim : int
            Dimension of the basis (number of modes).

        Returns
        -------
        IntegrationConfig
            Config with n_points = max(1000, 100 * dim).
        """
        n_points = max(1000, 100 * dim)
        return cls(n_points=n_points)


@dataclass
class ParallelConfig:
    """
    Configuration for parallel computation parameters.

    Parameters
    ----------
    enabled : bool
        Whether to use parallel computation.
    n_jobs : int
        Number of parallel jobs (-1 = all cores, 1 = no parallel).

    Examples
    --------
    >>> config = ParallelConfig()  # serial by default
    >>> config = ParallelConfig.all_cores()
    >>> config = ParallelConfig.cores(4)
    """

    enabled: bool = False
    n_jobs: int = -1

    def copy(self, **overrides) -> "ParallelConfig":
        """Create a copy with optional parameter overrides."""
        new_config = copy_module.copy(self)
        for key, value in overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return new_config

    @classmethod
    def all_cores(cls) -> "ParallelConfig":
        """Preset for parallel computation using all available cores."""
        return cls(enabled=True, n_jobs=-1)

    @classmethod
    def cores(cls, n: int) -> "ParallelConfig":
        """Preset for parallel computation using specific number of cores."""
        return cls(enabled=True, n_jobs=n)

    @classmethod
    def serial(cls) -> "ParallelConfig":
        """Preset for serial (non-parallel) computation."""
        return cls(enabled=False, n_jobs=1)


@dataclass
class LebesgueIntegrationConfig:
    """
    Hierarchical integration configuration for Lebesgue spaces.

    Allows different integration settings for:
    - Inner products (used by l2_inner_product, Gram matrices)
    - Dual operations (used by LinearFormKernel, to_dual/from_dual)
    - General operations (fallback for other integrations)

    Examples
    --------
    >>> config = LebesgueIntegrationConfig()
    >>> config.inner_product.n_points = 10000  \
    # High accuracy for Gram matrices
    """

    inner_product: IntegrationConfig = field(default_factory=IntegrationConfig)
    dual: IntegrationConfig = field(default_factory=IntegrationConfig)
    general: IntegrationConfig = field(default_factory=IntegrationConfig)

    @classmethod
    def from_single(
        cls, config: IntegrationConfig
    ) -> "LebesgueIntegrationConfig":
        """
        Create hierarchical config using same settings for all subsystems.
        """
        return cls(
            inner_product=config.copy(),
            dual=config.copy(),
            general=config.copy(),
        )

    @classmethod
    def high_accuracy_galerkin(cls) -> "LebesgueIntegrationConfig":
        """
        Preset optimized for accurate Galerkin matrix assembly.

        Uses high-accuracy integration for inner products and dual operations.
        """
        return cls(
            inner_product=IntegrationConfig(method="simpson", n_points=10000),
            dual=IntegrationConfig(method="simpson", n_points=10000),
            general=IntegrationConfig(),
        )

    @classmethod
    def adaptive_spectral(cls, dim: int) -> "LebesgueIntegrationConfig":
        """
        Preset for spectral methods with adaptive point selection.

        Scales integration points with basis dimension.

        Parameters
        ----------
        dim : int
            Number of basis functions.
        """
        n_points = max(1000, 100 * dim)
        return cls(
            inner_product=IntegrationConfig(n_points=n_points),
            dual=IntegrationConfig(n_points=n_points),
            general=IntegrationConfig(n_points=n_points),
        )


@dataclass
class LebesgueParallelConfig:
    """
    Hierarchical parallelization configuration for Lebesgue spaces.

    Allows different parallel settings for:
    - Inner products (used by l2_inner_product, Gram matrices)
    - Dual operations (used by LinearFormKernel, to_dual/from_dual)
    - General operations (fallback for other operations)

    Examples
    --------
    >>> config = LebesgueParallelConfig.parallel_dual(n_jobs=8)
    """

    inner_product: ParallelConfig = field(default_factory=ParallelConfig)
    dual: ParallelConfig = field(default_factory=ParallelConfig)
    general: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_single(cls, config: ParallelConfig) -> "LebesgueParallelConfig":
        """
        Create hierarchical config using same settings for all subsystems.
        """
        return cls(
            inner_product=config.copy(),
            dual=config.copy(),
            general=config.copy(),
        )

    @classmethod
    def parallel_dual(cls, n_jobs: int = -1) -> "LebesgueParallelConfig":
        """
        Preset with parallelization enabled only for dual operations.

        Parameters
        ----------
        n_jobs : int
            Number of cores for dual operations (-1 = all cores).
        """
        return cls(
            inner_product=ParallelConfig.serial(),
            dual=ParallelConfig(enabled=True, n_jobs=n_jobs),
            general=ParallelConfig.serial(),
        )

    @classmethod
    def all_parallel(cls, n_jobs: int = -1) -> "LebesgueParallelConfig":
        """
        Preset with parallelization enabled for all operations.

        Parameters
        ----------
        n_jobs : int
            Number of cores (-1 = all cores).
        """
        parallel = ParallelConfig(enabled=True, n_jobs=n_jobs)
        return cls.from_single(parallel)

    @classmethod
    def serial(cls) -> "LebesgueParallelConfig":
        """Preset with serial computation for all operations."""
        return cls.from_single(ParallelConfig.serial())
