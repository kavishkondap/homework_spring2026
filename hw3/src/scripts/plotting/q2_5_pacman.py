import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PLOT_X = "step"
PLOT_Ys = ["Train_EpisodeReturn", "Eval_AverageReturn"]

SAVE_NAME="q2_5_pacman.png"

ckpt_dir = "/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/MsPacman_dqn_sd1_20260310_062456"

fig = go.Figure()
df = pd.read_csv(Path(ckpt_dir) / "log.csv")

for plot_y in PLOT_Ys:
    df_temp = df[[plot_y, PLOT_X]][~df[plot_y].isna()]

    fig.add_trace(go.Scatter(x=df_temp[PLOT_X], y=df_temp[plot_y], mode="lines", name=plot_y))

# fig.add_hline(y=200,
#               line_dash="dash",
#                 line_color="red",
#                 annotation_text="y = 200",
#                 annotation_position="top left")
fig.update_layout(
    title="Q2.5 MsPacman - Eval vs. Train Return",
    xaxis_title=PLOT_X,
    yaxis_title="Return"
)

fig.write_image(SAVE_NAME)
# fig.show()