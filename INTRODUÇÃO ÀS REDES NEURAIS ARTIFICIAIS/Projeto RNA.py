# Carrega as bibliotecas
from keras import models
from keras import layers
# Instaciar a rede neural
network = models.Sequential()

# Adcionar uma camada totalmente conectada, com a função de ativação ReLU
network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))

# adciona uma segunda camada totalmente conectada com a função de ativação ReLU
network.add(layers.Dense(units=16, activation="relu"))

# adciona uma terceira camada totalmente conectada com a função de ativação sigmoid
network.add(layers.Dense(units=1, activation="sigmoid"))

# Compilar a rede neural
network.compile(loss="binary_crossentropy", # Cross-entropy
                optimizer="rmsprop", # Root Mean Square Propagation
                metrics=["accuracy"]) # Acurácia como metrica de desempenho