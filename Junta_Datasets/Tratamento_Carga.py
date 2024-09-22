#%% Importando Bibliotecas
import pandas as pd
#%% Leitura inicial dos dados
years = ["02", "03", "04", "05", "06", "07", "08", "09", 
           "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
           "20", "21", "22", "23", "24"]

path = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_2001.csv"
carga = pd.read_csv(path, delimiter=";", parse_dates=True)
carga

#%% Adição dos anos posteriores a 2001
for year in years:
    path_aux = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_20"+year+".csv"
    carga_aux = pd.read_csv(path_aux, delimiter=';', parse_dates=True)
    carga = pd.concat([carga, carga_aux])
carga.din_instante = pd.to_datetime(carga.din_instante, format="mixed")
carga

# %% Adicionando coluna que une Ano e Semana, que será usada na primeira hierarquia do agrupamento
serie_year = carga.din_instante.dt.year.astype(str)
serie_week = carga.din_instante.dt.isocalendar().week.astype(str)
carga["Ano_sem"] = serie_year + "_" + serie_week
carga_grouped = carga.drop(["id_subsistema", "din_instante"], axis=1).groupby(["Ano_sem", "nom_subsistema"], as_index=False).mean()
carga_grouped 


#%% Seleciona os dados do nordeste
carga_grouped_ne = carga_grouped.loc[carga_grouped.nom_subsistema =="Nordeste"]
carga_grouped_ne
#%% Exportando o dataset em um arquivo .csv
carga_grouped_ne.drop("nom_subsistema", axis=1).to_csv("Carga_Semanal_NE_2001_2024.csv")