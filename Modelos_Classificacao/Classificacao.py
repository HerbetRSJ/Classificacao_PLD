#%% IMPLEMENTAÇÃO DOS CLASSIFICADORES

#%% Importando bibliotecas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

average_metric = "weighted"
# %%
df = pd.read_csv("dataset_modelo.csv", index_col=0)
#df = df.iloc[:624]
df
#%%
df.columns
# %%
x_val = df.drop("Class", axis=1)
scaler = StandardScaler()
X_norm = scaler.fit_transform(x_val)
X_train, X_test, y_train, y_test = train_test_split(X_norm, df["Class"], test_size=0.3, random_state=42)

# %% CALCULA O PESO DAS CLASSES
class_weights = compute_class_weight("balanced", classes=np.unique(df["Class"]), y=y_train)
weights_dict = {class_label: weight for class_label, weight in zip(np.unique(df["Class"]), class_weights)}
weights_dict
#%%

clf = DecisionTreeClassifier(max_depth=8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")
cm = confusion_matrix(y_test, y_pred)
# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues")

# %%

clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")
cm = confusion_matrix(y_test, y_pred)
# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues")

# %%

model = MLPClassifier(hidden_layer_sizes=(100,5), 
                      activation="relu", 
                      solver='adam', 
                      alpha=0.001, 
                      max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.5f}")
cm = confusion_matrix(y_test, y_pred)
# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
# %% AA

# Criar o modelo
model = MLPClassifier(hidden_layer_sizes=(15,5), activation="relu", solver='adam', alpha=0.5, max_iter=10000)

# Ajustar o modelo aos dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.5f}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(include_values=True, cmap='Blues')



# %%
from sklearn.neighbors import KNeighborsClassifier

acuracias_knn = []

for i in range(20):
    neigh = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    # Avaliar a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    acuracias_knn.append(accuracy)
print(f'Acurácia média do KNN: {np.mean(acuracias_knn)}')

# %%
from sklearn.ensemble import RandomForestClassifier

# Inicializar o Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=500,max_depth=20, class_weight=weights_dict)

# Treinar o modelo
rf_classifier.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_classifier.predict(X_test)

# Avaliar a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Precision: {precision_score(y_test,y_pred, average=average_metric)}')
print(f'Recall: {recall_score(y_test,y_pred, average=average_metric)}')
print(f'F1: {f1_score(y_test,y_pred, average=average_metric)}')
print(f'Acc: {accuracy_score(y_test,y_pred)}')
#print(f'ROC AUC: {roc_auc_score(y_test,y_pred)}')

cm = confusion_matrix(y_test, y_pred)
# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues")
#%%
from sklearn.svm import SVC

clf = SVC(kernel='rbf')  # Pode-se alterar o kernel para 'rbf', 'poly', etc.

# Treinar o modelo
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar o desempenho
print(f'Acurácia: {accuracy_score(y_test, y_pred)}')

#%%
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
encoder = LabelEncoder()

classes = df["Class"].apply(lambda x: 0 if x == "muito_baixo" else
                            1 if x == "baixo" else
                            2 if x == "medio" else
                            3 if x == "alto" else
                            4)
X_train, X_test, y_train, y_test = train_test_split(X_norm, 
                                                    classes, 
                                                    test_size=0.3, 
                                                    random_state=42)

#%%
xg = XGBClassifier()

# Treinar o modelo
xg.fit(X_train, y_train)

# Fazer previsões
y_pred = xg.predict(X_test)

# Avaliar o desempenho
print(f'Acurácia do XGBoost: {accuracy_score(y_test, y_pred)}')

#%% 
y_train
#%%
from sklearn.model_selection import GridSearchCV

# Definir os parâmetros para tuning
param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [3, 5, 7],          
    'learning_rate': [0.01, 0.1, 0.2],  
    'subsample': [0.6, 0.8, 1.0],    
    'colsample_bytree': [0.6, 0.8, 1.0] 
}

grid_search = GridSearchCV(estimator=xg, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Melhores parâmetros: {grid_search.best_params_}')


best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy}')