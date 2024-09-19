#%%
import pandas as pd


path = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/balanco_energia_subsistema_ho/BALANCO_ENERGIA_SUBSISTEMA_2022.csv"
df = pd.read_csv(path, sep=";")
df
# %%
df["teste"] = df.val_gerhidraulica + df.val_gertermica + df.val_gereolica + df.val_gersolar - df.val_carga
df
# %%
