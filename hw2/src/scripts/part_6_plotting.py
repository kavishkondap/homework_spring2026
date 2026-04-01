import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PLOT_X = "Train_EnvstepsSoFar"
PLOT_Y = "Eval_AverageReturn"

SAVE_NAME="part_6_2.png"

ckpt_dirs = {
    "Default Hyperparams":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/InvertedPendulum-v4_pendulum_sd1_20260225_191503",
    # "Tuned Hyperparams":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/InvertedPendulum-v4_pendulum_5_sd1_20260225_193105",
    "Tuned Hyperparams":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/InvertedPendulum-v4_pendulum_6_sd1_20260225_212834"
}
# ckpt_dirs = {
#     "Standard":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_sd1_20260225_153621",
#     "RTG":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_rtg_sd1_20260225_152411",
#     "RTG & NA":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_rtg_na_sd1_20260225_182840",
#     "NA":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_na_sd1_20260225_152437"
# }

# ckpt_dirs = {
#     "NA":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_lb_na_sd1_20260225_152853",
#     "RTG & NA":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_lb_rtg_na_sd1_20260225_183226",
#     "RTG":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_lb_rtg_sd1_20260225_152806",
#     "Standard":"/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw2/exp/CartPole-v0_cartpole_lb_sd1_20260225_152641"
# }

fig = go.Figure()

for label, dir in ckpt_dirs.items():
    df = pd.read_csv(Path(dir) / "log.csv")

    fig.add_trace(go.Scatter(x=df[PLOT_X], y=df[PLOT_Y], mode="lines", name=label))

fig.update_layout(
    title="Pendulum Policy Hyperparameter Tuning",
    xaxis_title=PLOT_X,
    yaxis_title=PLOT_Y
)

fig.write_image(SAVE_NAME)
# fig.show()