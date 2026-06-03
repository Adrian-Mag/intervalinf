# Property Target Kernels - Mathematical Reference

This document explains the mathematical foundations of the property target kernels in `property_targets.py`.

## Real Spherical Harmonic Coefficient Mapping

The code maps real-harmonic component index `t` to complex spherical harmonic coefficients stored in pyshtools.

### Complex to Real Harmonic Conversion

Complex spherical harmonics $Y_{l,m}$ (with $m = -l, \ldots, l$) are converted to real spherical harmonics:

- **Zonal ($m=0$)**: $Y_{l,0}$ is already real
- **Cosine terms** (even real harmonics): $\propto Y_{l,m} + Y_{l,-m}$ → real part
- **Sine terms** (odd real harmonics): $\propto i(Y_{l,m} - Y_{l,-m})$ → imaginary part

### Storage Format

The pyshtools array `coeffs_array[2, lmax+1, lmax+1]` stores:
- `coeffs_array[0, l, m]` = real part of complex coefficient $Y_{l,m}$
- `coeffs_array[1, l, m]` = imaginary part of complex coefficient $Y_{l,m}$

### Index Mapping

The real-harmonic index $t$ ($0 \leq t \leq 2s$) maps as:

- **$t = 0$**: Zonal harmonic ($m=0$)
  - Returns `coeffs_array[0, s, 0]` (real part, already real)

- **$t$ odd** ($1, 3, 5, \ldots$): Cosine terms
  - $m = (t + 1) // 2$ (e.g., $t=1 \to m=1$, $t=3 \to m=2$)
  - Returns `coeffs_array[0, s, m]` (real part)

- **$t$ even** ($2, 4, 6, \ldots$): Sine terms
  - $m = t // 2$ (e.g., $t=2 \to m=1$, $t=4 \to m=2$)
  - Returns `coeffs_array[1, s, m]` (imaginary part)

This gives $2s+1$ real coefficients per degree $s$: 1 zonal + $s$ cosine + $s$ sine terms.

## Angular Bump Functions

### Gaussian Angular Bump

The angular bump is a Gaussian cap on the sphere centered at the north pole:

$$f(\theta) = \exp\!\left(-\frac{\theta^2}{2\sigma^2}\right)$$

Where:
- $\theta$ is the colatitude in radians
- $\sigma = \sigma_{\text{deg}} \cdot \pi/180$ is the angular half-width in radians

### Normalized Angular Bump

The normalized version ensures unit surface integral:

$$\int_{\Omega} f(\theta) \, d\Omega = 1$$

where $d\Omega = \sin(\theta) \, d\theta \, d\phi$ is the spherical surface element.

The normalization is computed numerically on a Driscoll-Healy (DH) grid:

$$\text{surface integral} = \sum f(\theta) \sin(\theta) \, \Delta\theta \, \Delta\phi$$

### Rotation to Target Location

The axisymmetric bump template is rotated to a target geographic location $(\text{lat}, \text{lon})$ using Euler angles:
- $\alpha = \text{lon}$ (rotation about z-axis)
- $\beta = 90^\circ - \text{lat}$ (tilt from north pole = colatitude)
- $\gamma = 0$

## Radial Bump Functions

### Gaussian Radial Bump

The radial bump is a Gaussian in radius:

$$h(r) = \exp\!\left(-\frac{(r - r_0)^2}{2\,\sigma_r^2}\right)$$

Where:
- $r$ is the radial coordinate (in km)
- $r_0$ is the center of the bump (in km)
- $\sigma_r$ is the half-width of the bump (in km)

### Normalized Radial Bump

The normalized version accounts for the spherical volume element:

$$\int h(r)\,r^2\,dr = 1$$

The normalization is computed numerically:

$$\text{volume integral} = \int h(r) \, r^2 \, dr$$

using trapezoidal integration on the radial grid.

## Property Target Families

### 1. Bulk Volumetric Averages (`BulkTarget`)

These compute weighted averages of radial parameters ($v_p$, $v_s$, $\rho$) over a localized region:

$$\mathcal{T}_i = \int h(r) \cdot b(\xi - \xi_0) \, m(r, \xi) \, r^2 dr \, d\Omega$$

Where:
- $h(r)$ is the radial bump (Gaussian in radius)
- $b(\xi - \xi_0)$ is the angular bump (Gaussian cap on sphere)
- $\xi_0 = (\text{lat}_0, \text{lon}_0)$ is the target location
- $m(r, \xi)$ is the model parameter field

The target is the product: $h(r) \cdot b(\xi - \xi_0)$.

### 2. CMB Topography Angular Bumps (`CMBTarget`)

These compute weighted CMB boundary displacement at specific locations:

$$\mathcal{T}_i = \int b(\xi - \xi_0) \, \sigma_{\text{CMB}}(\xi) \, d\Omega$$

Where:
- $b(\xi - \xi_0)$ is the angular bump (Gaussian cap on sphere)
- $\sigma_{\text{CMB}}(\xi)$ is the CMB topography field
- No radial component (topography is defined at the CMB boundary)

## Block Property Coefficients

For each spectral block $(s, t)$, the pre-computed angular coefficient $B_{st,i}$ for target $i$ is:

$$B_{st,i} = \text{extract\_st\_coeff}(\text{coeffs\_rot}_i, s, t)$$

Where $\text{coeffs\_rot}_i$ is the spherical harmonic expansion of target $i$'s angular bump rotated to its target location.

This allows efficient construction of property operators without recomputing spherical harmonic expansions for each block.
