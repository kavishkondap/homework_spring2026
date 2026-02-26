import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PLOT_X = "Train_EnvstepsSoFar"
PLOT_Y = "Baseline Loss"

SAVE_NAME="part_4_baseline_loss.png"

# ckpt_dirs = {
#     "Standard":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_sd1_20260225_153621",
#     "RTG":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_rtg_sd1_20260225_152411",
#     "RTG & NA":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_rtg_na_sd1_20260225_154105",
#     "NA":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_na_sd1_20260225_152437"
# }

ckpt_dirs = {
    "Baseline":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/HalfCheetah-v4_cheetah_baseline_sd185_20260225_175711",
}

fig = go.Figure()

for label, dir in ckpt_dirs.items():
    df = pd.read_csv(Path(dir) / "log.csv")

    fig.add_trace(go.Scatter(x=df[PLOT_X], y=df[PLOT_Y], mode="lines", name=label))

fig.update_layout(
    title="Large Batch Size Experiments",
    xaxis_title=PLOT_X,
    yaxis_title=PLOT_Y
)

fig.write_image(SAVE_NAME)
# fig.show()