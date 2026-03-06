import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from scipy.interpolate import RectBivariateSpline

# Fix for the PDF Type 3 font issue from your previous file
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Load CSV data
df_acmpc = pd.read_csv("results/acmpc/action_penalty/results/run_20260219203108/mass_sweep_acmpc.csv")
df_ffn = pd.read_csv("results/ffn/action_penalty/results/run_20260223130705/mass_sweep_ffn.csv")

# Build meshgrid from the data
b1_vals = np.sort(df_acmpc["blue1_mass"].unique())
b2_vals = np.sort(df_acmpc["blue2_mass"].unique())

# Pivot to 2D arrays (rows=blue2, cols=blue1)
wr_acmpc = df_acmpc.pivot(index="blue2_mass", columns="blue1_mass", values="blue_win_rate").values
wr_ffn = df_ffn.pivot(index="blue2_mass", columns="blue1_mass", values="blue_win_rate").values
wr_diff = wr_acmpc - wr_ffn

# Interpolate to finer grid for smooth band edges
b1_fine = np.linspace(b1_vals[0], b1_vals[-1], 200)
b2_fine = np.linspace(b2_vals[0], b2_vals[-1], 200)
B1f, B2f = np.meshgrid(b1_fine, b2_fine)

win_fine_acmpc = RectBivariateSpline(b2_vals, b1_vals, wr_acmpc)(b2_fine, b1_fine)
win_fine_ffn = RectBivariateSpline(b2_vals, b1_vals, wr_ffn)(b2_fine, b1_fine)
diff_fine = RectBivariateSpline(b2_vals, b1_vals, wr_diff)(b2_fine, b1_fine)

# Common axis setup
nominal = 0.0406 * 1000
ymin, ymax = b2_vals[0] * 1000, b2_vals[-1] * 1000
xmin, xmax = b1_vals[0] * 1000, b1_vals[-1] * 1000
ticks = np.arange(40.6 - 10, 40.6 + 11, 4)
ticks = ticks[(ticks >= xmin - 0.5) & (ticks <= xmax + 0.5)]

# Create the 3-panel plot
fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

# ACMPC and FFN panels
for ax, data, title in zip(axes[:2], [win_fine_acmpc, win_fine_ffn], ['MA-ACMPC', 'MLP']):
    im = ax.contourf(B1f * 1000, B2f * 1000, data * 100, levels=np.linspace(0, 100, 11), cmap='viridis')
    ax.set_title(f'{title}: Evader Win Rate', fontsize=28, fontweight='bold', pad=18)
    ax.set_xlabel('Evader 1 Mass [g]', fontsize=24)
    ax.set_ylabel('Evader 2 Mass [g]', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.plot([nominal, nominal], [ymin, nominal], color='k', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.plot([xmin, nominal], [nominal, nominal], color='k', linestyle='--', linewidth=1.5, alpha=0.8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Win Rate [%]', fontsize=22)
    cb.ax.tick_params(labelsize=18)

# Difference panel
max_diff = np.max(np.abs(wr_diff)) * 100
diff_limit = np.ceil(max_diff / 10) * 10  # round up to nearest 10
im3 = axes[2].contourf(B1f * 1000, B2f * 1000, diff_fine * 100, levels=np.linspace(-diff_limit, diff_limit, 21), cmap='RdBu_r')
axes[2].contour(B1f * 1000, B2f * 1000, diff_fine * 100, levels=[0], colors='black', linewidths=2.5, linestyles='--')
axes[2].set_title('Delta (MA-ACMPC - MLP)', fontsize=28, fontweight='bold', pad=18)
axes[2].set_xlabel('Evader 1 Mass [g]', fontsize=24)
axes[2].set_ylabel('Evader 2 Mass [g]', fontsize=24)
axes[2].tick_params(labelsize=20)
axes[2].set_xticks(ticks)
axes[2].set_yticks(ticks)
axes[2].plot([nominal, nominal], [ymin, nominal], color='k', linestyle='--', linewidth=1.5, alpha=0.8)
axes[2].plot([xmin, nominal], [nominal, nominal], color='k', linestyle='--', linewidth=1.5, alpha=0.8)
cb3 = fig.colorbar(im3, ax=axes[2])
cb3.set_label('$\Delta$ Win Rate [%]', fontsize=22)
cb3.ax.tick_params(labelsize=18)

plt.savefig('results/mass_sweep_comparison.pdf', dpi=300)
plt.show()
