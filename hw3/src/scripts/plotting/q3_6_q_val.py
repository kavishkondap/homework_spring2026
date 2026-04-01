import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PLOT_X = "step"
PLOT_Y = "q_values"

SAVE_NAME="q3_6_q_val.png"

ckpt_dirs = {
    "Single Q":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/Hopper-v4_sac_singleq_sd1_20260311_132318",
    "Double-Q Clipped":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/Hopper-v4_sac_clipq_sd1_20260311_132406",
}
fig = go.Figure()

for label, dir in ckpt_dirs.items():
    df = pd.read_csv(Path(dir) / "log.csv")
    df = df[[PLOT_X, PLOT_Y]][~df[PLOT_Y].isna()]

    fig.add_trace(go.Scatter(x=df[PLOT_X], y=df[PLOT_Y], mode="lines", name=label))

fig.update_layout(
    title="Q3.6 Q Values (Single Q vs. Clipped Double Q)",
    xaxis_title=PLOT_X,
    yaxis_title=PLOT_Y
)

fig.write_image(SAVE_NAME)
# fig.show()