import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#Nos permite cargar los datos de entrenamiento y test
"""
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)
"""

#Nos permite visualizar las imágenes de los números manuscritos
"""
def show_image(imagen, title):
    plt.figure()
    plt.suptitle(title)
    plt.imshow(imagen, cmap = "Greys")
    plt.show()
    
for i in range(3):
    title = "Mostrando imagen X_train[" + str(i) + "]"
    title = title + " -- Y_train[" + str(i) + "] = " + str(Y_train[i])
    show_image(X_train[i], title)
"""

#Nos permite visualizar los valores de los píxeles de una imagen
"""
def plot_X(X, title, fila, columna):
    plt.title(title)
    plt.plot(X)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.show()

fila = 5
columna = 6

features_fila_col = X_train[:, fila, columna]
print(len(np.unique(features_fila_col)))

title = "Valores en (" + str(fila) + ", " + str(columna) + ")"
plot_X(features_fila_col, title, fila, columna)
"""




##############################--Aqui empieza el código de la práctica--##############################

def load_MNIST_for_adaboost():
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0

    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

class DecisionStump:
    ## Constructor de clase, con número de características
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.rand()
        self.polarity = np.random.choice([-1, 1])
        return self

    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        # Si la característica que comprueba este clasificador es mayor que el umbral y la polaridad es 1
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase)
        # Si no, devolver -1 (no pertenece a la clase)
        predictions = np.ones(X.shape[0])
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions

class Adaboost:
    ## Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T
        self.A = A
        self.classifiers = []
        return self
    
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose = False):
        # Obtener el número de observaciones y de características por observación de X
        n_observations, n_features = X.shape

        # Iniciar pesos de las observaciones a 1/n_observaciones
        weights = np.ones(n_observations) / n_observations

        # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        for t in range(self.T):
            """
            if verbose:
                print("Entrenando clasificador débil " + str(t+1) + " de " + str(self.T))
            # Iniciar lista de clasificadores débiles
            classifiers = []
            # Iniciar lista de predicciones ponderadas de los clasificadores débiles
            predictions = np.zeros(n_observations)

            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
            for a in range(self.A):
                # Crear un nuevo clasificador débil aleatorio
                classifier = DecisionStump(n_features)
                # Calcular predicciones de ese clasificador para todas las observaciones
                classifier_predictions = classifier.predict(X)
                # Calcular el error: comparar predicciones con los valores deseados
                error = np.sum(weights[classifier_predictions != Y])
                # Actualizar mejor clasificador hasta el momento: el que tenga menor error
                if error < best_error:
                    best_classifier = classifier
                    best_error = error
                    best_predictions = classifier_predictions
                # Calcular el valor de alfa y las predicciones del mejor clasificador débil
                alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
                predictions += alpha * best_predictions
                # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
                weights *= np.exp(-alpha * Y * best_predictions)
                # Normalizar a 1 los pesos
                weights /= np.sum(weights)
                # Guardar el clasificador en la lista de clasificadores de Adaboost
                classifiers.append(best_classifier)
            self.classifiers.append(classifiers)
            """
        # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
        # Crear un nuevo clasificador débil aleatorio
        # Calcular predicciones de ese clasificador para todas las observaciones
        # Calcular el error: comparar predicciones con los valores deseados
        # y acumular los pesos de las observaciones mal clasificadas
        # Actualizar mejor clasificador hasta el momento: el que tenga menor error
        # Calcular el valor de alfa y las predicciones del mejor clasificador débil
        # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
        # Normalizar a 1 los pesos
        # Guardar el clasificador en la lista de clasificadores de Adaboost
    
    ## Método para obtener una predicción con el clasificador fuerte Adaboost
    def predict(self, X):
        # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        predictions = np.array([classifier.predict(X) * classifier.alpha for classifier in self.classifiers])
        # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        return np.sign(np.sum(predictions, axis=0))