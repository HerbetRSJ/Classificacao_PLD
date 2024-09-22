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

model = MLPClassifier(hidden_layer_sizes=(100,), 
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
model = MLPClassifier(hidden_layer_sizes=(15,5), activation="logistic", solver='adam', alpha=0.01, max_iter=10000)

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
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    # Avaliar a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    acuracias_knn.append(accuracy)
print(f'Acurácia média do KNN: {np.mean(acuracias_knn)}')
neigh.predict_proba(X_test)

# %%
from sklearn.ensemble import RandomForestClassifier

# Inicializar o Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=300,max_depth=16, class_weight=weights_dict)

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
df.Class.value_counts()