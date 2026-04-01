import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PLOT_X = "step"
PLOT_Y = "Eval_AverageReturn"

SAVE_NAME="q2_5_lunar_lander.png"

ckpt_dirs = {
    "Lunar Lander Return":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/LunarLander-v2_dqn_sd1_20260309_225910",
}
fig = go.Figure()

for label, dir in ckpt_dirs.items():
    df = pd.read_csv(Path(dir) / "log.csv")
    df = df[["Eval_AverageReturn", "step"]][~df["Eval_AverageReturn"].isna()]

    fig.add_trace(go.Scatter(x=df[PLOT_X], y=df[PLOT_Y], mode="lines", name=label))

fig.add_hline(y=200,
              line_dash="dash",
                line_color="red",
                annotation_text="y = 200",
                annotation_position="top left")
fig.update_layout(
    title="Q2.5 Lunar Lander Eval Average Return Across Training",
    xaxis_title=PLOT_X,
    yaxis_title=PLOT_Y
)

fig.write_image(SAVE_NAME)
# fig.show()