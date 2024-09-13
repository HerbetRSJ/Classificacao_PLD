# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
path = "Historico_do_Preco_Medio_Semanal_-_30_de_junho_de_2001_a_9_de_agosto_de_2024.xls"
pld = pd.read_excel(path)
pld.dtypes
#%%
pld["Ano_sem"] = pld.ANO.astype(str) + "_" + pld.DATA_INICIO.dt.isocalendar().week.astype(str)
pld
# %%
pld_grouped = pld.drop(["ANO", "MES", "SEMANA", "DATA_INICIO", "DATA_FIM"], 
                       axis=1).groupby(["Ano_sem"], as_index=False).mean()
pld_grouped

# %% 
pld_grouped[["Ano_sem", "NORDESTE"]].to_csv("PLD_NE_Semanal_2001_2024.csv")
