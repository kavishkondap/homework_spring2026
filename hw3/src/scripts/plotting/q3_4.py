import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PLOT_X = "step"
PLOT_Y = "Eval_AverageReturn"

SAVE_NAME="q3_4.png"

ckpt_dirs = {
    "Cheetah":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/HalfCheetah-v4_sac_sd1_20260311_194533",
}
fig = go.Figure()

for label, dir in ckpt_dirs.items():
    df = pd.read_csv(Path(dir) / "log.csv")
    df = df[["Eval_AverageReturn", "step"]][~df["Eval_AverageReturn"].isna()]

    fig.add_trace(go.Scatter(x=df[PLOT_X], y=df[PLOT_Y], mode="lines", name=label))

fig.update_layout(
    title="Q3.4 Eval Average Return Across Training",
    xaxis_title=PLOT_X,
    yaxis_title=PLOT_Y
)

fig.write_image(SAVE_NAME)
# fig.show()