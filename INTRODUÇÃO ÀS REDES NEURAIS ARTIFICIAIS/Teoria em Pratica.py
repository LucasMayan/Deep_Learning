# Carregar as bibliotecas
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Resetar o gerador de números randômicos
np.random.seed(0)

# Definir o número de atributos (features)
number_of_features = 14

# Instanciar a rede neural
network = models.Sequential()

# Adicionar a primeira camada, com a função de ativação ReLU
network.add(layers.Dense(units=32, activation="relu", 
                         input_shape=(number_of_features,)))

# Adicionar a segunda camada, com a função de ativação ReLU
network.add(layers.Dense(units=64, activation="relu"))

# Adicionar a terceira camada, com a função de ativação ReLU
network.add(layers.Dense(units=32, activation="relu"))

# Adicionar a quarta camada, com a função de ativação sigmoid
network.add(layers.Dense(units=1, activation="sigmoid"))

# Compilar a rede neural
network.compile(loss="binary_crossentropy", # função de perda cross-entropy
                optimizer="rmsprop", # função de otimização Root Mean Square Propagation
                metrics=["accuracy"]) # Usar a acurácia como métrica de desempenho
