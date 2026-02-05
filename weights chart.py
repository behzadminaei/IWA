import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set style for scientific publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Data
labels = ['C1', 'C2', 'C3', 'C4', 'C5']
mean_he = [0.099558, 0.076582, 0.059664, 0.068866, 0.054836]
weights = [0.150973, 0.197937, 0.231066, 0.192927, 0.227098]

# Number of variables
num_vars = len(labels)

# Angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Close the data loop
mean_he += mean_he[:1]
weights += weights[:1]

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

# Scientific color palette (accessible and publication-ready)
color_he = '#2E86AB'  # Professional blue
color_weights = '#A23B72'  # Professional purple/magenta

# Plot Mean Hyper-Entropy with enhanced styling
ax.plot(angles, mean_he, color=color_he, linewidth=2.5, 
        label='Mean Hyper-Entropy (He)', marker='o', markersize=8, 
        markerfacecolor='white', markeredgewidth=2, markeredgecolor=color_he,
        zorder=3)
ax.fill(angles, mean_he, color=color_he, alpha=0.15, zorder=1)

# Plot Weights with enhanced styling
ax.plot(angles, weights, color=color_weights, linewidth=2.5, 
        label='Criteria Weights (w)', marker='s', markersize=8,
        markerfacecolor='white', markeredgewidth=2, markeredgecolor=color_weights,
        zorder=3)
ax.fill(angles, weights, color=color_weights, alpha=0.15, zorder=1)

# Set radial limits with better range
max_value = max(max(mean_he), max(weights))
ax.set_ylim(0, max_value * 1.15)

# Add radial grid lines with better styling
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Customize radial grid
ax.set_rticks(np.linspace(0, max_value, 6))
ax.set_rlabel_position(22.5)
ax.tick_params(colors='#333333', labelsize=9)
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, color='#888888')

# Set angular labels with better positioning
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')

# Add value annotations on the plot
for angle, he_val, weight_val in zip(angles[:-1], mean_he[:-1], weights[:-1]):
    # Annotate Hyper-Entropy values
    ax.text(angle, he_val + max_value * 0.05, f'{he_val:.4f}', 
            ha='center', va='bottom', fontsize=8, color=color_he, fontweight='bold')
    # Annotate Weight values
    ax.text(angle, weight_val + max_value * 0.05, f'{weight_val:.4f}', 
            ha='center', va='top', fontsize=8, color=color_weights, fontweight='bold')

# Enhanced title
ax.set_title('Criteria Weights vs. Mean Hyper-Entropy\n(Radar Chart Analysis)', 
             fontsize=16, fontweight='bold', pad=30, color='#1a1a1a')

# Professional legend
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
                   frameon=True, fancybox=True, shadow=True,
                   framealpha=0.95, edgecolor='#333333', facecolor='white')
legend.get_frame().set_linewidth(1.2)

# Add subtle background circle for depth
circle = plt.Circle((0, 0), max_value, transform=ax.transData._b, 
                    fill=False, edgecolor='#e0e0e0', linewidth=1, alpha=0.3)
ax.add_patch(circle)

# Improve overall appearance
ax.spines['polar'].set_visible(False)
ax.set_facecolor('#fafafa')

# Add text box with summary statistics
stats_text = f'Max Weight: {max(weights[:-1]):.4f}\nMin Weight: {min(weights[:-1]):.4f}\nMax He: {max(mean_he[:-1]):.4f}\nMin He: {min(mean_he[:-1]):.4f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
        alpha=0.8, edgecolor='#cccccc', linewidth=1),
        family='monospace')

plt.tight_layout()
plt.savefig('weights_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
