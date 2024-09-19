#%% IMPLEMENTAÇÃO DOS CLASSIFICADORES

#%% Importando bibliotecas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

# %%
df = pd.read_csv("dataset_modelo.csv", index_col=0)
df
# %%
x_val = df.drop("Class", axis=1)
scaler = StandardScaler()
X_norm = scaler.fit_transform(x_val)
#%%
X_train, X_test, y_train, y_test = train_test_split(x_val, df["Class"], test_size=0.3)

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
X_train, X_test, y_train, y_test = train_test_split(x_val, df["Class"], test_size=0.3)

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
X_train, X_test, y_train, y_test = train_test_split(X_norm, df["Class"], test_size=0.3)

model = MLPClassifier(hidden_layer_sizes=(10,5), activation="relu", solver='adam', alpha=0.001, max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")
cm = confusion_matrix(y_test, y_pred)
# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
# %%
from imblearn.under_sampling import RandomUnderSampler

X_train, X_test, y_train, y_test = train_test_split(X_norm, df["Class"], test_size=0.3)
undersample = RandomUnderSampler(sampling_strategy='auto')
X_train, y_train = undersample.fit_resample(X_train, y_train)

# Criar o modelo
model = MLPClassifier(hidden_layer_sizes=(10,5), activation="logistic", solver='adam', alpha=0.001, max_iter=10000)

# Ajustar o modelo aos dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(include_values=True, cmap='Blues')