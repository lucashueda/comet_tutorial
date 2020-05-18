# Must be de first
from comet_ml import Experiment

# Importando as libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score

data = pd.read_csv("data/data.csv")

random_state = 42

# Instanciando um modelo de árvore de decisão
params = {
    'max_depth': 2,
    'random_state': random_state
}

model = DecisionTreeClassifier(**params)

experiment = Experiment(api_key="kDS2masAuJAetDECsRadUAKBQ",
                        project_name="sada", workspace="lucashueda")
model.fit(data[['x1','x2']], data['y'])
experiment.end()

y_pred = model.predict(data[['x1','x2']])

# Plotando na tela o gráfico de decisão do modelo
plt.figure(figsize=(5, 3))
plt.scatter(data['x1'], data['x2'], c=y_pred)
plt.title("Classificação da árvore de decisão")
plt.show()

acc = accuracy_score(data['y'], y_pred)
precision = precision_score(data['y'],y_pred, average= 'macro')
print(f"Acuracia de {acc} e precisao de {precision}")

# Adicionando log de métricas
experiment.log_metric('Acurácia', acc)
experiment.log_metric('Precisão', precision)

# Adicionando log de hparams
experiment.log_parameters(params)

