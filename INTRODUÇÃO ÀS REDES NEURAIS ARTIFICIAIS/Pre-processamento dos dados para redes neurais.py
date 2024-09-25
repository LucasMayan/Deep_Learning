# carrega as bibliotecas
from sklearn import preprocessing
import numpy as np

# cria um atributo
features = np.array([[-100.1, 3240.1],
                     [-200.2, -234.1],
                     [5000.5, 150.1],
                     [6000.6, -125.1],
                     [9000.9, -673.1]])

print(features)

# cria o objeto scaler
scaler = preprocessing.StandardScaler()

# transforma o atributo
features_standardized = scaler.fit_transform(features)

# mostra o atributo
print(features_standardized)