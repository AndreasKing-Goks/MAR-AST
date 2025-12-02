from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42    # Use TrueType instead of Type 3
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial' #'DejaVu Sans'  # Or: 'Arial', 'Helvetica'
import matplotlib.pyplot as plt

from utils.get_path import get_rl_csv_path

# === Load data ===
model_name = "AST-train_2025-11-15_03-22-41_dd72"
loss_path = Path(get_rl_csv_path(root=ROOT, model_name=model_name, csv_name="loss.csv"))
reward_path = Path(get_rl_csv_path(root=ROOT, model_name=model_name, csv_name="reward.csv"))

print(loss_path)
print(reward_path)

loss_df   = pd.read_csv(loss_path)     # columns: Wall time, Step, Value
reward_df = pd.read_csv(reward_path)   # columns: Wall time, Step, Value

# === Create side-by-side plots ===
fig, axes = plt.subplots(
    1, 2,
    figsize=(6.0, 2.3),  # tweak for your conference template
    constrained_layout=True
)

# --- Left: Actor loss ---
axes[0].plot(loss_df["Step"], loss_df["Value"])
axes[0].set_xlabel("Training step")
axes[0].set_ylabel("Actor loss")
axes[0].grid(True)

# --- Right: Rollout episode reward ---
axes[1].plot(reward_df["Step"], reward_df["Value"])
axes[1].set_xlabel("Training step")
axes[1].set_ylabel("Rollout episode reward")
axes[1].grid(True)

# Make ticks a bit smaller for paper style
for ax in axes:
    ax.tick_params(labelsize=8)

# Save to file (change as needed)
plt.savefig("actor_loss_and_reward.pdf", bbox_inches="tight")
# plt.savefig("actor_loss_and_reward.png", dpi=300, bbox_inches="tight")

plt.show()
