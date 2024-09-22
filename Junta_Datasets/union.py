#%%
import pandas as pd
#%% Importando os datasets
ear = pd.read_csv("EAR_Semanal_NE_2001_2024.csv", index_col=0)
ena = pd.read_csv("ENA_Semanal_NE_2001_2024.csv", index_col=0)
carga = pd.read_csv("Carga_Semanal_NE_2001_2024.csv", index_col=0)
geracao = pd.read_csv("Geracao_Semanal_NE_2001_2024.csv", index_col=0)
pld = pd.read_csv("PLD_NE_Semanal_2001_2024.csv", index_col=0)


#%% Une os datasets a partir da coluna "Ano_semana". Uniao do tipo 'left'
df_merged = ear.merge(ena, on="Ano_sem").merge(carga, on="Ano_sem").merge(geracao, on="Ano_sem").merge(pld, on="Ano_sem") 
df_merged
#%%
df_merged.to_csv("dataset_final.csv")

#%% 
df_merged.isna().sum()