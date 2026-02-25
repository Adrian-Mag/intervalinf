# Prior Sensitivity Study — Summary

All cases use identical vs, rho, σ₀, σ₁ priors. Only the vp prior is varied.

| Case | s | ls | ov | offset | RMSE | MAE | Coverage(2σ) | Mean σ_post | Mean σ_prior | Unc. Reduction | Data Misfit | Min Eig. | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 6.0 | 30 | 31.622776601683793 | 0.0 | 0.035881 | 0.022045 | 0.5 | 0.009667 | 0.038288 | 0.747533 | 502.484547 | 0.0 | 59.5 |
| Over-smooth | 12.0 | 30 | 31.622776601683793 | 0.0 | 0.038243 | 0.024254 | 0.25 | 0.005081 | 0.028566 | 0.822147 | 544.780072 | -0.0 | 51.12 |
| Under-smooth | 3.0 | 10 | 31.622776601683793 | 0.0 | 0.015124 | 0.010967 | 1.0 | 0.024656 | 0.075978 | 0.675489 | 474.367344 | 7.86e-06 | 63.68 |
| Shifted-mean | 6.0 | 30 | 31.622776601683793 | 3.0 | 0.430514 | 0.271295 | 0.0 | 0.009667 | 0.038288 | 0.747533 | 10074.900557 | 0.0 | 59.75 |


## Metric Definitions

| Metric | Definition |
| --- | --- |
| RMSE | √( (1/Nₚ) Σᵢ (p̃ᵢ − pᵢ*)² ) |
| MAE | (1/Nₚ) Σᵢ |p̃ᵢ − pᵢ*| |
| Coverage(2σ) | Fraction of i with |p̃ᵢ − pᵢ*| ≤ 2σᵢ |
| Mean σ_post | (1/Nₚ) Σᵢ σᵢ  (posterior std) |
| Mean σ_prior | (1/Nₚ) Σᵢ σᵢ⁰ (prior std pushed through T) |
| Unc. Reduction | 1 − Mean σ_post / Mean σ_prior |
| Data Misfit | ‖G(m̃) − d̃‖₂ |
| Min Eig. | λ_min of property posterior covariance (stability) |
