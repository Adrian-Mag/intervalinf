# intervalinf Migration Status

**Last Updated:** December 30, 2025
**Current Phase:** Phase 5 COMPLETED, Phase 6 next
**Tests Passing:** 270

## Project Overview

We are extracting the `interval` module from `pygeoinf` into a standalone package called `intervalinf`. This makes the interval-specific functionality independent while maintaining compatibility with `pygeoinf` for core abstractions (LinearOperator, HilbertSpace, etc.).

### Source and Target Locations
- **Source:** `/disks/data/PhD/Inferences/pygeoinf/pygeoinf/interval/`
- **Target:** `/disks/data/PhD/Inferences/intervalinf/`
- **Package location:** `/disks/data/PhD/Inferences/intervalinf/intervalinf/`

### Key Design Decisions
1. **pygeoinf as dependency:** Operators inherit from `pygeoinf.linear_operators.LinearOperator` and spaces from `pygeoinf.hilbert_space.HilbertSpace`
2. **Providers created in Phase 3:** Although originally planned for Phase 4, providers were created during Phase 3 because operators depend on them
3. **Basis initialization:** Lebesgue spaces use `basis=None` by default; basis providers connected in Phase 4
4. **Function decoupling:** Functions can now be standalone (domain-only) or attached to a space; providers support both modes

---

## Migration Plan (6 Phases)

### ✅ Phase 1: Core Module (COMPLETED)
**Files created:**
- `intervalinf/core/__init__.py`
- `intervalinf/core/domain.py` - `IntervalDomain` class
- `intervalinf/core/boundary.py` - `BoundaryConditions` class
- `intervalinf/core/config.py` - `IntegrationConfig`, `ParallelConfig`, `GalerkinConfig`
- `intervalinf/core/functions.py` - `Function` class (~670 lines)

**Tests:** 145 passing

---

### ✅ Phase 2: Spaces Module (COMPLETED)
**Files created:**
- `intervalinf/spaces/__init__.py`
- `intervalinf/spaces/lebesgue.py` - `Lebesgue`, `PartitionedLebesgueSpace`, direct sum support (~1000 lines)
- `intervalinf/spaces/sobolev.py` - `Sobolev` space with Laplacian-based norm
- `intervalinf/spaces/forms.py` - `LinearFormKernel` for dual representations

**Tests:** 207 passing (62 new)

---

### ✅ Phase 3: Operators Module (COMPLETED)
**Files created:**

#### Operators (`intervalinf/operators/`)
| File | Classes | Description |
|------|---------|-------------|
| `base.py` | `SpectralOperator` | Abstract base class for spectral operators |
| `laplacian.py` | `Laplacian`, `InverseLaplacian` | ~490 lines, spectral/FD/FEM methods |
| `gradient.py` | `Gradient` | Finite difference gradient operator |
| `bessel.py` | `BesselSobolev`, `BesselSobolevInverse` | Bessel potential (Matérn), fast transforms |
| `sola.py` | `SOLAOperator` | Kernel-based integration operator |
| `radial.py` | `RadialLaplacian`, `InverseRadialLaplacian`, `RadialLaplacianEigenvalueProvider`, `RadialLaplacianSpectrumProvider` | For spherical coordinates |
| `spectral_helpers.py` | Helper functions | `build_eigenfunction_expansion`, `validate_eigenvalue`, etc. |
| `_impl/fem_solvers.py` | `GeneralFEMSolver` | FEM solver for inverse Laplacian |
| `_impl/fast_spectral.py` | Fast transforms | DST/DCT/DFT coefficient computation |

#### Providers (`intervalinf/providers/`)
| File | Classes | Description |
|------|---------|-------------|
| `base.py` | `FunctionProvider`, `IndexedFunctionProvider`, `RestrictedFunctionProvider`, `ParametricFunctionProvider`, `RandomFunctionProvider`, `NullFunctionProvider` | Abstract provider classes |
| `functions.py` | `SineFunctionProvider`, `CosineFunctionProvider`, `FourierFunctionProvider`, `MixedDNFunctionProvider`, `MixedNDFunctionProvider`, `RobinFunctionProvider`, `HatFunctionProvider` | Basis function providers |
| `spectrum.py` | `EigenvalueProvider`, `SineEigenvalueProvider`, `CosineEigenvalueProvider`, `FourierEigenvalueProvider`, `ZeroEigenvalueProvider`, `CustomEigenvalueProvider`, `LaplacianEigenvalueProvider`, `SpectrumProvider`, `LaplacianSpectrumProvider` | Eigenvalue/spectrum providers |
| `radial.py` | `RadialLaplacianDirichletProvider`, `RadialLaplacianNeumannProvider`, `RadialLaplacianDDProvider`, `RadialLaplacianDNProvider`, `RadialLaplacianNDProvider`, `RadialLaplacianNNProvider` | Radial eigenfunction providers |

#### Utilities (`intervalinf/utils/`)
| File | Classes | Description |
|------|---------|-------------|
| `robin_utils.py` | `RobinRootFinder` | Root-finding for Robin BC eigenvalues |

**Tests:** 231 passing (24 new operator tests)

---

### ✅ Phase 4: Providers Integration (COMPLETED)
**Goal:** Connect the providers to the Lebesgue/Sobolev spaces and restructure provider hierarchy.

**Changes made:**
1. Created hierarchical provider structure under `providers/functions/`:
   - `trigonometric.py` - Sine, Cosine, Fourier, MixedDN, MixedND, Robin
   - `fem.py` - HatFunctionProvider, SplineFunctionProvider
   - `smooth.py` - BumpFunctionProvider, BumpFunctionGradientProvider
   - `wavelets.py` - WaveletFunctionProvider
   - `step.py` - BoxCarFunctionProvider, DiscontinuousFunctionProvider
   - `data.py` - KernelProvider, NormalModesProvider
2. Updated `create_basis_provider()` in `lebesgue.py` to use intervalinf providers
3. Added `space_or_domain` support to all providers (can create standalone functions)
4. Decoupled `Function` class from spaces - functions can now be standalone or attached

**Tests:** 270 passing (including 20 standalone provider tests)

---

### ✅ Phase 5: Sampling Module (COMPLETED)
**Goal:** Migrate the KL sampling utilities.

**Source file migrated:**
- `pygeoinf/interval/KL_sampler.py` → `intervalinf/sampling/kl_sampler.py`

**Files created:**
| File | Classes | Description |
|------|---------|-------------|
| `sampling/__init__.py` | Module exports | Exports KLSampler, TruncationInfo |
| `sampling/kl_sampler.py` | `KLSampler`, `TruncationInfo` | Spectral (KL) sampling for Gaussian measures (~370 lines) |

**Features:**
- Truncated KL expansion for Gaussian measures
- Support for both Lebesgue and Sobolev (mass-weighted) spaces
- Covariance factor operator L with C ≈ L L*
- Variance function computation
- Energy-based or explicit mode truncation

**Tests:** 270 passing

---

### 🔲 Phase 6: Integration & Cleanup (NEXT)
**Goal:** Final integration, documentation, and cleanup

**Tasks:**
- Update main `intervalinf/__init__.py` with all public exports
- Add comprehensive docstrings
- Create usage examples
- Final test coverage review
- Remove any remaining pygeoinf dependencies (except core abstractions)

---

## Current Package Structure

```
intervalinf/
├── pyproject.toml
├── README.md
├── MIGRATION_STATUS.md  (this file)
├── intervalinf/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── domain.py
│   │   ├── boundary.py
│   │   ├── config.py
│   │   └── functions.py
│   ├── spaces/
│   │   ├── __init__.py
│   │   ├── lebesgue.py
│   │   ├── sobolev.py
│   │   └── forms.py
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── laplacian.py
│   │   ├── gradient.py
│   │   ├── bessel.py
│   │   ├── sola.py
│   │   ├── radial.py
│   │   ├── spectral_helpers.py
│   │   └── _impl/
│   │       ├── __init__.py
│   │       ├── fem_solvers.py
│   │       └── fast_spectral.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── spectrum.py
│   │   ├── radial.py
│   │   └── functions/
│   │       ├── __init__.py
│   │       ├── trigonometric.py
│   │       ├── fem.py
│   │       ├── smooth.py
│   │       ├── wavelets.py
│   │       ├── step.py
│   │       └── data.py
│   ├── sampling/
│   │   ├── __init__.py
│   │   └── kl_sampler.py
│   └── utils/
│       ├── __init__.py
│       └── robin_utils.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── core/
    │   ├── __init__.py
    │   ├── test_boundary.py
    │   ├── test_config.py
    │   ├── test_domain.py
    │   └── test_functions.py
    ├── spaces/
    │   ├── __init__.py
    │   ├── test_forms.py
    │   ├── test_lebesgue.py
    │   └── test_sobolev.py
    ├── operators/
    │   ├── __init__.py
    │   └── test_operators.py
    └── providers/
        ├── __init__.py
        └── test_standalone_providers.py
```

---

## Important Technical Notes

### 1. Function class accepts IntervalDomain directly
The `Function` class in `core/functions.py` was updated to handle `IntervalDomain` directly as the `space` argument (not just Lebesgue/Sobolev spaces). The `function_domain` property checks `isinstance(self.space, IntervalDomain)` first.

### 2. pygeoinf imports in operators
Operators import from pygeoinf:
```python
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.hilbert_space import EuclideanSpace
```
This is by design - pygeoinf provides the abstract base classes.

### 3. Lebesgue basis initialization
Currently `Lebesgue(dim, domain, basis='sine')` will fail unless pygeoinf is installed because the fallback import fails. Phase 4 needs to fix this by connecting to `intervalinf.providers`.

### 4. Test commands
```bash
cd /disks/data/PhD/Inferences/intervalinf
python -m pytest tests/ -v --tb=short  # All tests
python -m pytest tests/operators/ -v   # Just operator tests
```

### 5. Python environment
- Conda environment: `inferences2`
- Python version: 3.11

---

## Git Status
- Repository: `/disks/data/PhD/Inferences/intervalinf`
- Branch: `main`
- Last commit: "Phase 3: Add operators module with providers"
- 21 files changed, 5503 insertions

---

## Immediate Next Steps for Phase 4

1. **Read** `intervalinf/spaces/lebesgue.py` lines 850-930 to understand current `create_basis_provider()` implementation

2. **Update** `create_basis_provider()` to import from `intervalinf.providers`:
   ```python
   from intervalinf.providers import (
       SineFunctionProvider,
       CosineFunctionProvider,
       FourierFunctionProvider,
       # etc.
   )
   ```

3. **Test** that `Lebesgue(100, domain, basis='sine')` works without pygeoinf

4. **Add tests** for basis functionality in `tests/spaces/test_lebesgue.py`

5. **Commit** Phase 4 changes

---

## Reference: Source File Locations in pygeoinf

| intervalinf module | pygeoinf source |
|-------------------|-----------------|
| `core/domain.py` | `interval/domain.py` |
| `core/boundary.py` | `interval/boundary_conditions.py` |
| `core/functions.py` | `interval/functions.py` |
| `spaces/lebesgue.py` | `interval/lebesgue_space.py` |
| `spaces/sobolev.py` | `interval/sobolev_space.py` |
| `operators/*.py` | `interval/operators/*.py` |
| `providers/*.py` | `interval/function_providers/*.py` |
| `sampling/` | `interval/sampling/` (to be migrated) |
