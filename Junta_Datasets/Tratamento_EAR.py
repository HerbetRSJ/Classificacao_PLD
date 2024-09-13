#%% Importando Bibliotecas
import pandas as pd
#%% Leitura inicial dos dados
years = ["02", "03", "04", "05", "06", "07", "08", "09", 
           "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
           "20", "21", "22", "23", "24"]

path = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_2001.csv"
ear = pd.read_csv(path, delimiter=";", parse_dates=True)
ear

#%% Adição dos anos posteriores a 2001
for year in years:
    path_aux = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_20"+year+".csv"
    ear_aux = pd.read_csv(path_aux, delimiter=';', parse_dates=True)
    ear = pd.concat([ear, ear_aux])
ear.ear_data = pd.to_datetime(ear.ear_data, format="mixed")
ear

# %% Adicionando coluna que une Ano e Semana, que será usada na primeira hierarquia do agrupamento
serie_year = ear.ear_data.dt.year.astype(str)
serie_week = ear.ear_data.dt.isocalendar().week.astype(str)
ear["Ano_sem"] = serie_year + "_" + serie_week

# Foram utilizadas operaçoes diferentes de agrupamento, dada a característida de cada atributo
ear_grouped = ear.drop(["id_subsistema", "ear_data"], axis=1).groupby(["Ano_sem", "nom_subsistema"], as_index=False).agg({"ear_max_subsistema": "max",
                                                                                                                          "ear_verif_subsistema_mwmes": "sum",
                                                                                                                          "ear_verif_subsistema_percentual": "mean"})
ear_grouped
#%% Seleciona os dados do nordeste
ear_grouped_ne = ear_grouped.loc[ear_grouped.nom_subsistema =="NORDESTE"]
ear_grouped_ne
#%% Exportando o dataset em um arquivo .csv
ear_grouped_ne.drop("nom_subsistema", axis=1).to_csv("EAR_Semanal_NE_2001_2024.csv")