import numpy as np
import tensorflow as tf
from tensorflow import keras 
"""Oui, la méthode fit() est nécessaire pour l'apprentissage d'un réseau de neurones Keras. 
Cette méthode est utilisée pour entraîner un modèle en utilisant un ensemble de données 
d'entraînement, et elle ajuste les poids du modèle en utilisant l'algorithme d'optimisation spécifié dans le modèle.
un exemple de code Python qui utilise la méthode fit() pour entraîner un modèle de réseau de neurones Keras :"""



# Définir l'architecture du modèle
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compiler le modèle avec une fonction de perte et un optimiseur
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Charger les données MNIST qui n'ont rien à voir avec un puissance 4 (juste la bonne dimension)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Prétraiter les données
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Entraîner le modèle
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
print(history)

"""Dans cet exemple, le modèle est défini comme un réseau de neurones avec une couche cachée de 64 neurones avec une fonction d'activation ReLU,
 suivie d'une couche de sortie avec une fonction d'activation softmax. 
 Le modèle est compilé avec une fonction de perte de categorical_crossentropy et un optimiseur adam. 
 Ensuite, les données MNIST sont chargées et prétraitées en normalisant les valeurs de pixel et en transformant les étiquettes en vecteurs binaires. 
 Enfin, le modèle est entraîné avec la méthode fit() en utilisant les données d'entraînement, avec 5 époques et une taille de lot de 64. 
 La méthode fit() renvoie un objet history qui contient des informations sur la performance du modèle pendant l'entraînement.


"""

 
#Pour tester le modèle de réseau de neurones que j'ai montré ci-dessus, vous pouvez utiliser le code suivant :
"""
Ce code charge les données de test MNIST, prétraite les données en normalisant les valeurs de pixel 
et en transformant les étiquettes en vecteurs binaires. 
Enfin, le code utilise le modèle pour faire des prédictions sur les données de test et
calcule la précision en comparant les prédictions avec les étiquettes réelles. 
La précision est ensuite affichée à l'écran.


"""
# Charger les données MNIST
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Prétraiter les données de test
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_test = keras.utils.to_categorical(y_test, 10)


# Faire des prédictions sur les données de test
predictions = model.predict(x_test)

# Convertir les prédictions en étiquettes de classe
predicted_labels = np.argmax(predictions, axis=1)

# Calculer la précision sur les données de test
accuracy = np.mean(predicted_labels == np.argmax(y_test, axis=1))
print('Accuracy:', accuracy)


