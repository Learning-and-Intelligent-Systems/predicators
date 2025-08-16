import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Font settings for consistency
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 14

# Colors from your existing palette
ours_color = "#10a37f"
vila_fewshot_color = "#d8b88f"

# Data from the table
data = {
    ("Cleanup", "Ours"):            [100, 100, 100, 100, 100],
    ("Cleanup", "ViLa-fewshot"):    [100, 100, 66,  0,   33],
    ("Juice", "Ours"):              [66,  66,  100, 66,  33],
    ("Juice", "ViLa-fewshot"):      [0,   66,  100, 0,   0],
}

axes_labels = ["New obj.", "New vis.", "More obj.", "Novel goal 1", "Novel goal 2"]

# Convert to DataFrame
rows = []
for (task, approach), vals in data.items():
    for axis, v in zip(axes_labels, vals):
        rows.append({
            "Task": task,
            "Approach": approach,
            "Axis": axis,
            "Success %": v
        })
df = pd.DataFrame(rows)
df["Axis"] = pd.Categorical(df["Axis"], categories=axes_labels, ordered=True)

# Create vertical stack of horizontal bar plots
fig, axes = plt.subplots(2, 1, figsize=(4, 4.8), sharex=True)

tasks = ["Cleanup", "Juice"]
for ax, task in zip(axes, tasks):
    d = df[df["Task"] == task]
    sns.barplot(
        data=d, y="Axis", x="Success %",
        hue="Approach",
        palette=[ours_color, vila_fewshot_color],
        ax=ax, capsize=0.1, orient='h'
    )

    ax.set_title(task, fontsize=14)
    ax.set_xlabel("% success (3 trials)", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, 110)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', labelsize=11)

# Legend above the top plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
axes[0].get_legend().remove()
axes[1].get_legend().remove()

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
