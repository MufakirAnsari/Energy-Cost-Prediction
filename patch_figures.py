"""Quick patch to fix figure titles and labels in step_20_fix_weaknesses.py"""
import re

filepath = "/home/ansari/Desktop/Energy/V2/step_20_fix_weaknesses.py"

with open(filepath, "r") as f:
    content = f.read()

# Fix 1: W2 suptitle - remove "Figure W2: " prefix
content = content.replace(
    'fig.suptitle("Figure W2: The Sharpness',
    'fig.suptitle("The Sharpness'
)
content = content.replace(
    'QRF achieves target coverage with moderate width; CQR either under-covers or over-corrects"',
    'QRF achieves target coverage; CQR under-covers or over-corrects"'
)

# Fix 2: W3 suptitle - remove "Figure W3: " prefix
content = content.replace(
    'fig.suptitle("Figure W3: MC Dropout',
    'fig.suptitle("MC Dropout'
)

# Fix 3: W4 suptitle - remove "Figure W4: The " prefix
content = content.replace(
    'fig.suptitle("Figure W4: The MAE',
    'fig.suptitle("MAE'
)

# Fix 4: W4 - replace annotations with legend-based labels
# Remove the annotate calls for W4 and add label= to scatter
old_w4_scatter = '''            ax.scatter(mae, rmse, s=size, color=color, zorder=5,
                      edgecolors="black", linewidths=0.8,
                      marker="*" if i == len(configs) - 1 else "o")
            ax.annotate(c, (mae, rmse), fontsize=7, ha="left", va="bottom",
                       xytext=(5, 3), textcoords="offset points")'''

new_w4_scatter = '''            ax.scatter(mae, rmse, s=size, color=color, zorder=5,
                      edgecolors="black", linewidths=0.8,
                      marker="*" if i == len(configs) - 1 else "o",
                      label=c)'''

content = content.replace(old_w4_scatter, new_w4_scatter)

# Fix 5: W4 - add legend after title
old_w4_title = '''        ax.set_title(f"{market.upper()}: Ensemble Composition Trade-off", fontweight="bold")

        # Add annotation box'''

new_w4_title = '''        ax.set_title(f"{market.upper()}: Ensemble Composition Trade-off", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
                 borderpad=0.4, title="Composition", title_fontsize=8)
'''

content = content.replace(old_w4_title, new_w4_title)

# Fix 6: Make all fig sizes and tight_layout consistent
content = content.replace(
    'figsize=(13, 5.5))\n\n    for ax, market in zip(axes, ["pjm", "ercot"]):\n        ens_path',
    'figsize=(14, 6))\n\n    for ax, market in zip(axes, ["pjm", "ercot"]):\n        ens_path'
)

# Fix 7: All tight_layout calls should use rect to avoid suptitle clipping
# Only replace the ones that don't already have rect
lines = content.split('\n')
new_lines = []
for i, line in enumerate(lines):
    if 'fig.tight_layout()' in line and 'rect' not in line:
        new_lines.append(line.replace('fig.tight_layout()', 'fig.tight_layout(rect=[0, 0, 1, 0.90])'))
    else:
        new_lines.append(line)
content = '\n'.join(new_lines)

# Fix 8: Bump all suptitle fontsize from 10 to 11
content = content.replace("fontsize=10, fontweight=\"bold\")", "fontsize=11, fontweight=\"bold\")")

with open(filepath, "w") as f:
    f.write(content)

print("Patch applied successfully!")
print("Changes made:")
print("  - Removed 'Figure W2/W3/W4' prefixes from all suptitles")
print("  - Replaced W4 text annotations with legend entries")
print("  - Added legend to W4 scatter plots")  
print("  - Added rect spacing to tight_layout for suptitle clearance")
print("  - Bumped suptitle fontsize to 11")
