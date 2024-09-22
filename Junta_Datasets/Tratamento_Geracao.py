# -*- coding: utf-8 -*-
#%% Importando Bibliotecas
import pandas as pd

years = ["02", "03", "04", "05", "06", "07", "08", "09",
           "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
           "20", "21"]
years2 = ["22", "23", "24"]
months = ["_01", "_02", "_03", "_04", "_05", "_06", "_07", "_08", "_09", "_10", "_11", "_12"]

path = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/GERACAO_USINA_2001.csv"
df = pd.read_csv(path, delimiter=";", parse_dates=True)

#%% Adição dos anos posteriores a 2001
for year in years:
    path_aux = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/GERACAO_USINA_20"+year+".csv"
    ger_aux = pd.read_csv(path_aux, delimiter=';', parse_dates=True)
    print(f"Ano {year} Adicionado!")
    df = pd.concat([df, ger_aux])
#%%
for year in years2:
    for month in months:
        path_aux = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/GERACAO_USINA-2_20"+year+month+".csv"
        ger_aux = pd.read_csv(path_aux, delimiter=';', parse_dates=True)
        df = pd.concat([df, ger_aux])
    print(f"Ano {year} Adicionado!")  
#%% 
df = df.loc[df.nom_subsistema == "NORDESTE"]
df
#%%
df
#%%
df.columns

columns_to_drop = ['id_subsistema', 'nom_subsistema', 'id_estado', 'nom_estado', 'cod_modalidadeoperacao','nom_tipocombustivel', 'nom_usina', 'ceg']
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned.head(30)

# Cria uma coluna de para cada tipo de geração
df_pivot = df_cleaned.pivot_table(index="din_instante",columns="nom_tipousina", aggfunc="sum", values="val_geracao")
df_pivot = df_pivot.reset_index()
df_pivot



# Confere de a conversão deu certo
df_pivot.din_instante = pd.to_datetime(df_pivot.din_instante)
df_pivot.dtypes

# Remove as horas para facilitar o agrupamento
df_pivot.din_instante = pd.to_datetime(df_pivot["din_instante"].dt.date)

# Realiza o agrupamento por dia
geracao = df_pivot.groupby(df_pivot["din_instante"], as_index=False).sum()
geracao

# Realiza o agrupamento semanal
serie_year = geracao.din_instante.dt.year.astype(str)
serie_week = geracao.din_instante.dt.isocalendar().week.astype(str)
geracao["Ano_sem"] = serie_year + "_" + serie_week
geracao_grouped = geracao.drop(["din_instante"], axis=1).groupby(["Ano_sem"], as_index=False).mean()
geracao_grouped
# %%
geracao_grouped.to_csv("Geracao_Semanal_NE_2001_2024.csv")
# %%
