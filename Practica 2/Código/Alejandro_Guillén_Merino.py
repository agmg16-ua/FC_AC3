import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#Nos permite 

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
"""
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
        return self

    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        # Si la característica que comprueba este clasificador es mayor que el umbral y la polaridad es 1
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase)
        # Si no, devolver -1 (no pertenece a la clase)

class Adaboost:
    ## Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        return self
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose = False):
        # Obtener el número de observaciones y de características por observación de X
        # Iniciar pesos de las observaciones a 1/n_observaciones
        # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
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
        # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo       