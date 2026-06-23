"""Render the LakePilot two-semester project timeline as a Gantt chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NAVY, TEAL, RED, AMBER, BLUE = "#1B3139", "#17A2A2", "#FF3621", "#EF9F27", "#185FA5"

# (task, start_week, duration_weeks, colour, phase)
tasks = [
    ("Literature review & problem framing",      0,  3, BLUE,  "S1"),
    ("LakeGym v5 environment (Spark + Iceberg)",  2,  4, TEAL,  "S1"),
    ("Reward design & workload modelling",        4,  3, TEAL,  "S1"),
    ("Heuristic baselines & data collection",     6,  3, AMBER, "S1"),
    ("Meta-controller + specialist agents",       8,  4, NAVY,  "S1"),
    ("Interim report & design freeze",           11,  2, RED,   "S1"),
    ("Offline pre-training (AWR/PPO/DDQN)",       13,  4, TEAL,  "S2"),
    ("Frozen & adaptive benchmarking",           16,  3, BLUE,  "S2"),
    ("Ablation & cost analysis",                 18,  3, AMBER, "S2"),
    ("Results analysis & figures",               20,  3, NAVY,  "S2"),
    ("Final report & paper write-up",            22,  4, RED,   "S2"),
]

fig, ax = plt.subplots(figsize=(11, 5.2))
for i, (name, start, dur, color, _) in enumerate(tasks):
    y = len(tasks) - 1 - i
    ax.barh(y, dur, left=start, height=0.55, color=color, alpha=0.92,
            edgecolor="white", linewidth=1.2)
    ax.text(start + dur + 0.2, y, f"{dur}w", va="center", fontsize=8, color="#5F6F75")

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=9)
ax.set_xlabel("Project week", fontsize=10)
ax.set_xlim(0, 27)
ax.axvline(13, color="#9FB8BF", linestyle="--", linewidth=1)
ax.text(6.5, len(tasks) - 0.3, "Semester 1", ha="center", fontsize=10, color=NAVY, weight="bold")
ax.text(20, len(tasks) - 0.3, "Semester 2", ha="center", fontsize=10, color=NAVY, weight="bold")
ax.grid(axis="x", color="#E3E1DA", linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#9FB8BF")
plt.title("LakePilot — Two-Semester Project Timeline (Gantt)", fontsize=12, color=NAVY, weight="bold")
plt.tight_layout()
plt.savefig("docs/diagrams/gantt.png", dpi=160, bbox_inches="tight", transparent=False, facecolor="white")
print("gantt.png written")
