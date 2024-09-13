#%% Importando Bibliotecas
import pandas as pd
#%% Leitura inicial dos dados
years = ["02", "03", "04", "05", "06", "07", "08", "09", 
           "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
           "20", "21", "22", "23", "24"]

path = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_2001.csv"
ena = pd.read_csv(path, delimiter=";", parse_dates=True)
ena

#%% Adição dos anos posteriores a 2001
for year in years:
    path_aux = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_20"+year+".csv"
    ena_aux = pd.read_csv(path_aux, delimiter=';', parse_dates=True)
    ena = pd.concat([ena, ena_aux])
ena.ena_data = pd.to_datetime(ena.ena_data, format="mixed")
ena

# %% Adicionando coluna que une Ano e Semana, que será usada na primeira hierarquia do agrupamento
serie_year = ena.ena_data.dt.year.astype(str)
serie_week = ena.ena_data.dt.isocalendar().week.astype(str)
ena["Ano_sem"] = serie_year + "_" + serie_week

# Foram utilizadas operaçoes diferentes de agrupamento, dada a característida de cada atributo
ena_grouped = ena.drop(["id_subsistema", "ena_data"], axis=1).groupby(["Ano_sem", "nom_subsistema"], as_index=False).agg({"ena_bruta_regiao_mwmed": "sum",
                                                                                                                          "ena_bruta_regiao_percentualmlt": "mean",
                                                                                                                          "ena_armazenavel_regiao_mwmed": "sum",
                                                                                                                          "ena_armazenavel_regiao_percentualmlt": "mean"})
ena_grouped 

#%% Seleciona os dados do nordeste
ena_grouped_ne = ena_grouped.loc[ena_grouped.nom_subsistema =="NORDESTE"]
ena_grouped_ne
#%% Exportando o dataset em um arquivo .csv
ena_grouped_ne.drop("nom_subsistema", axis=1).to_csv("ENA_Semanal_NE_2001_2024.csv")
