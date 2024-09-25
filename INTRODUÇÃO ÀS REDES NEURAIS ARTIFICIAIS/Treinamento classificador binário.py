# Carrega as bibliotecas
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import warnings
warnings.filterwarnings('ignore')

# Resetar o gerador de numeros randomicos
np.random.seed(0)

# definir o numero de atributos (features)
number_of_feature = 1000

# Carregar os dados (conjunto de dados de avaliação de filmes)
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_feature)

# Converter o formato dos dados para uma matriz de features
tokenizer = Tokenizer(num_words=number_of_feature)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")  

# Instanciar a rede neural
network = models.Sequential()

# Adicionar uma camada totalmente conectada, com a função de ativação ReLU
network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_feature,)))

# Adicionar uma segunda camada totalmente conectada com a função de ativação ReLU
network.add(layers.Dense(units=16, activation="relu"))

# Adicionar uma terceira camada totalmente conectada com a função de ativação sigmoid
network.add(layers.Dense(units=1, activation="sigmoid"))

# Compilar a rede neural
network.compile(loss="binary_crossentropy", # Cross-entropy
                optimizer="rmsprop", # Root Mean Square Propagation
                metrics=["accuracy"]) # Acurácia como métrica de desempenho
