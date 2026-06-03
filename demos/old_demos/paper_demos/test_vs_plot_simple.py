"""
Simple test for the combined vs plotting logic.
Tests the key changes: combining IC and mantle grids with outer core shading.
"""
import numpy as np
import matplotlib.pyplot as plt

# Load saved data
data = np.load('tuner_result.npz')
print("Loaded saved data")

# Extract vs data
r_vs_IC = data['r_vs_IC']
r_vs_M = data['r_vs_M']
mean_vs_IC = data['mean_vs_IC']
mean_vs_M = data['mean_vs_M']
probe_r_vs_IC = data['probe_r_vs_IC']
probe_r_vs_M = data['probe_r_vs_M']
std_vs_IC = data['std_vs_IC']
std_vs_M = data['std_vs_M']

# Domain bounds
ICB = 1217.5
CMB = 3480.0

# Combine the two vs grids into a single continuous grid (this is the key change)
r_vs_combined = np.concatenate([r_vs_IC, r_vs_M])
mean_vs_combined = np.concatenate([mean_vs_IC, mean_vs_M])

# Interpolate stds onto combined grid
std_vs_IC_dense = np.interp(r_vs_IC, probe_r_vs_IC, std_vs_IC)
std_vs_M_dense = np.interp(r_vs_M, probe_r_vs_M, std_vs_M)
std_vs_combined = np.concatenate([std_vs_IC_dense, std_vs_M_dense])

# Create test plot
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Plot combined vs
ax.plot(mean_vs_combined, r_vs_combined, color='forestgreen', lw=1.5, label='mean')
ax.fill_betweenx(
    r_vs_combined,
    mean_vs_combined - std_vs_combined,
    mean_vs_combined + std_vs_combined,
    alpha=0.3, color='forestgreen', label='±1σ',
)

# Shade outer core region (ICB to CMB) - this is the key feature
ax.axhspan(ICB, CMB, color='gray', alpha=0.3, label='outer core')

ax.set_ylabel('radius (km)')
ax.set_xlabel('δvs')
ax.set_title('Combined vs with outer core shading (test)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('test_vs_simple.png', dpi=150)
print("Saved test plot to test_vs_simple.png")
plt.close(fig)

print("Test successful! The plot shows:")
print("- Combined inner core and mantle vs on one panel")
print(f"- Gray shaded outer core region from {ICB} to {CMB} km")
