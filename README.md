# Aide_IA_Keras

Ce code définit l'architecture du modèle en utilisant la classe Sequential de Keras. Dans cet exemple, le modèle comporte deux couches, une couche cachée avec 64 neurones et une fonction d'activation ReLU, et une couche de sortie avec 10 neurones et une fonction d'activation softmax.

Le modèle est ensuite compilé avec une fonction de perte de categorical_crossentropy et un optimiseur adam. Cette étape configure le modèle pour l'entraînement en spécifiant la façon dont il doit être optimisé.

Les données d'entraînement sont ensuite chargées et prétraitées. Dans cet exemple, il s'agit des données MNIST, un ensemble de 60 000 images de chiffres manuscrits de 28x28 pixels. Les données sont normalisées en divisant les valeurs de pixel par 255 et les étiquettes sont transformées en vecteurs binaires à l'aide de la fonction to_categorical de Keras.

Enfin, le modèle est entraîné en utilisant la méthode fit() de Keras. Cette méthode ajuste les poids du modèle en utilisant les données d'entraînement, en effectuant des itérations sur un certain nombre d'époques et en ajustant les poids à chaque itération en utilisant l'algorithme d'optimisation spécifié lors de la compilation du modèle.

Ensuite, le code teste le modèle en chargeant les données de test MNIST, en prétraitant les données et en utilisant le modèle pour faire des prédictions. La précision du modèle est calculée en comparant les prédictions avec les étiquettes réelles.

Enfin, le modèle est sauvegardé dans un fichier au format h5 en utilisant la méthode save() de Keras.
