import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

PLOT_X = "step"
PLOT_Y = "Eval_AverageReturn"

SAVE_NAME="q3_5.png"

ckpt_dirs = {
    "Cheetah Fixed Temperature":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/HalfCheetah-v4_sac_sd1_20260311_194533",
    "Cheetah Auto-tuned Temperature":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw3/exp/HalfCheetah-v4_sac_autotune_sd1_20260311_205129"
}
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Eval Average Return (Fixed vs. Auto-tune Temp.)", "Auto-tuned Temp.")
)

for label, dir in ckpt_dirs.items():
    df = pd.read_csv(Path(dir) / "log.csv")
    df = df[["Eval_AverageReturn", "step"]][~df["Eval_AverageReturn"].isna()]

    fig.add_trace(go.Scatter(x=df[PLOT_X], y=df[PLOT_Y], mode="lines", name=label), row=1, col = 1)

auto_tune_df = pd.read_csv(Path(ckpt_dirs["Cheetah Auto-tuned Temperature"]) / "log.csv")

auto_tune_df = auto_tune_df[["temperature", "step"]][~auto_tune_df["temperature"].isna()]
fig.add_trace(go.Scatter(x=auto_tune_df["step"], y=auto_tune_df["temperature"], mode="lines", name="Auto-tune Temp."), row=2, col = 1)
fig.update_layout(
    title="Q3.5 Training Results",
    xaxis_title=PLOT_X,
    yaxis_title=PLOT_Y
)

fig.write_image(SAVE_NAME)
# fig.show()