import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. YOUR EXACT DATA FROM THE BENCHMARK LOG
# ==========================================
labels = ['Baseline\n(Blunt POMCP)', 'Proposed\n(Heuristic POMCP)']

# Panel A: Success Rate
success_rates = [42.0, 90.0]

# Panel B: Time-to-Capture (Frames)
frames = [1089.6, 778.1]
frames_std = [270.7, 290.7] # These are the standard deviations for the error bars

# Panel C: Planning Latency (ms)
latency = [74.28, 189.12]
latency_std = [16.27, 55.00] # Standard deviations

# ==========================================
# 2. PLOT SETTINGS FOR IEEE WIDTH
# ==========================================
# 14x5 is the perfect aspect ratio for a wide, three-panel, double-column figure.
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Academic color palette
# Grey = Baseline, Blue/Green/Red = Your different metrics
color_baseline = '#a0a0a0'
colors_proposed = ['#4c72b0', '#55a868', '#c44e52'] # Blue, Green, Red

# --- PANEL 1: SUCCESS RATE (%) ---
# Title case formatting for axis labels and titles
bars1 = axes[0].bar(labels, success_rates, color=[color_baseline, colors_proposed[0]], edgecolor='black', width=0.5)
axes[0].set_title('Capture Success Rate (%)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Success Rate (%)', fontsize=11)
axes[0].set_ylim(0, 110) # Give some space at the top for labels

# Add percentage labels on top of the bars
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 2, 
                 f"{yval}%", ha='center', va='bottom', 
                 fontweight='bold', fontsize=11)

# --- PANEL 2: TIME-TO-CAPTURE (FRAMES) ---
# We include 'yerr' to automatically draw the error bars
bars2 = axes[1].bar(labels, frames, yerr=frames_std, capsize=6, # capsize adds little caps to the error lines
                    color=[color_baseline, colors_proposed[1]], edgecolor='black', width=0.5)
axes[1].set_title('Avg. Time-to-Capture (Frames)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Time (Frames)', fontsize=11)
axes[1].set_ylim(0, 1600)

# Add the mean number inside the bars for clarity
for i, bar in enumerate(bars2):
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval / 2, 
                 f"{yval}", ha='center', va='center', 
                 fontweight='bold', fontsize=11, color='white')

# --- PANEL 3: PLANNING LATENCY (MS) ---
bars3 = axes[2].bar(labels, latency, yerr=latency_std, capsize=6, 
                    color=[color_baseline, colors_proposed[2]], edgecolor='black', width=0.5)
axes[2].set_title('Avg. Planning Latency (ms)', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Latency (ms)', fontsize=11)
axes[2].set_ylim(0, 280)

# Add the mean number inside the bars
for i, bar in enumerate(bars3):
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2, yval / 2, 
                 f"{yval}", ha='center', va='center', 
                 fontweight='bold', fontsize=11, color='white')

# ==========================================
# 3. GLOBAL FORMATTING FOR ALL AXES
# ==========================================
for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0) # Add background gridlines
    ax.set_axisbelow(True) # Ensure bars are drawn on top of the gridlines
    ax.tick_params(axis='x', labelsize=11) # Increase X-axis label size for the paper

# === SAVE THE HIGH-RES FILE ===
plt.tight_layout()
plt.savefig('ablation_metrics.png', dpi=300, bbox_inches='tight') # dpi=300 is publication quality
print("Successfully generated 'ablation_metrics.png'!")
plt.show()