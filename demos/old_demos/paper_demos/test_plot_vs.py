"""
Quick test script for the modified plot_block_posterior function.
Uses saved data from tuner_result.npz to test the vs visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Load saved data
data = np.load('tuner_result.npz')
print("Loaded saved data with keys:", list(data.keys()))

# Extract data
r_vp = data['r_vp']
r_vs_IC = data['r_vs_IC']
r_vs_M = data['r_vs_M']
r_rho = data['r_rho']
mean_vp = data['mean_vp']
mean_vs_IC = data['mean_vs_IC']
mean_vs_M = data['mean_vs_M']
mean_rho = data['mean_rho']
probe_r_vp = data['probe_r_vp']
probe_r_vs_IC = data['probe_r_vs_IC']
probe_r_vs_M = data['probe_r_vs_M']
probe_r_rho = data['probe_r_rho']
std_vp = data['std_vp']
std_vs_IC = data['std_vs_IC']
std_vs_M = data['std_vs_M']
std_rho = data['std_rho']
sigma_1_mean = data['sigma_1_mean']
sigma_1_std = data['sigma_1_std']

# Create mock objects that mimic the structure expected by plot_block_posterior
@dataclass(frozen=True, order=True)
class BlockIndex:
    s: int
    t: int

@dataclass
class RadialSpecs:
    earth_radius_km: float = 6371.0
    icb_radius_km: float = 1217.5
    cmb_radius_km: float = 3480.0

# Mock function class
class MockFunction:
    def __init__(self, r_vals, mean_vals):
        self.r_vals = r_vals
        self.mean_vals = mean_vals
    
    def __call__(self, r):
        # Simple interpolation
        return np.interp(r, self.r_vals, self.mean_vals)

# Create mock posterior expectation structure
block = BlockIndex(s=2, t=0)
specs = RadialSpecs()

# Mock the expectation structure: [[f_vp, [f_vs_IC, f_vs_M], f_rho], [sigma_0_arr, sigma_1_arr]]
f_vp = MockFunction(r_vp, mean_vp)
f_vs_IC = MockFunction(r_vs_IC, mean_vs_IC)
f_vs_M = MockFunction(r_vs_M, mean_vs_M)
f_rho = MockFunction(r_rho, mean_rho)
mock_expectation = [
    [f_vp, [f_vs_IC, f_vs_M], f_rho],
    [np.array([0.0]), np.array([sigma_1_mean])]
]

# Mock posterior
class MockPosterior:
    def __init__(self, expectation):
        self.expectation = expectation

mock_posterior = MockPosterior(mock_expectation)

# Mock forward_dict entry
class MockSpace:
    def __init__(self, domain_a, domain_b):
        self.function_domain = type('obj', (object,), {'a': domain_a, 'b': domain_b})()

class MockDirectSum:
    def __init__(self, subspaces):
        self.subspaces = subspaces

# Create mock model spaces
M_vp_space = MockSpace(0.0, 6371.0)
M_vs_IC_space = MockSpace(0.0, 1217.5)
M_vs_M_space = MockSpace(3480.0, 6371.0)
M_rho_space = MockSpace(0.0, 6371.0)

M_vs_space = MockDirectSum([M_vs_IC_space, M_vs_M_space])
M_functions = MockDirectSum([M_vp_space, M_vs_space, M_rho_space])
M_euclidean = MockDirectSum([MockSpace(0, 1), MockSpace(0, 1)])
M_st = MockDirectSum([M_functions, M_euclidean])

# Mock forward_dict entry
mock_forward_entry = (None, None, M_st, None)
forward_dict = {block: mock_forward_entry}

# Now test the plotting function
import sys
sys.path.insert(0, 'visualization')
from full_spectrum_viz import plot_block_posterior

print("Testing plot_block_posterior with saved data...")
fig = plot_block_posterior(
    block=block,
    posterior=mock_posterior,
    forward_dict=forward_dict,
    specs=specs,
    n_grid=200,
    n_probes=20,
)

print("Plot generated successfully!")
fig.savefig('test_vs_plot.png', dpi=150)
print("Saved to test_vs_plot.png")
plt.close(fig)
