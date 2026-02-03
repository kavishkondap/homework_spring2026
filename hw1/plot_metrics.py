import plotly.express as px
import pandas as pd

CSV_PATH = "/Users/kavish/Desktop/KAVISH/Berkeley/Sophomore Year/cs185/homework_spring2026/hw1/exp/seed_42_20260203_122136/log.csv"
POLICY_TYPE = "flow"

df = pd.read_csv(CSV_PATH)[1:] #exclude first row
df_loss = df.loc[:, ["loss", "step"]][~df["loss"].isna()]
df_reward = df.loc[:, ["eval/mean_reward", "step"]][~df["eval/mean_reward"].isna()]

fig = px.line(df_loss, x="step", y="loss", title=f"Training Loss ({POLICY_TYPE} Policy)")
fig.write_image(f"{POLICY_TYPE}_loss.png")

fig = px.line(df_reward, x="step", y="eval/mean_reward", title=f"Mean Reward ({POLICY_TYPE} Policy)", text="eval/mean_reward")
fig.update_traces(
    texttemplate="%{y:.3f}",
    textposition="top center"
)
fig.write_image(f"{POLICY_TYPE}_mean_reward.png")