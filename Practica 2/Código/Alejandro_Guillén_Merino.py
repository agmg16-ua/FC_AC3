import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time as time
import random as random

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


##DECISION STUMP
class DecisionStump:
    ## Constructor de clase, con número de características
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.caracteristica = random.randint(0, n_features - 1)
        self.umbral = np.random.rand()
        self.polaridad = np.random.choice([-1, 1])


    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        # Si la característica que comprueba este clasificador es mayor que el umbral y la polaridad es 1
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase)
        # Si no, devolver -1 (no pertenece a la clase)
        """
        predicciones = []
        for x in (X):
            
            if (x[self.caracteristica] > self.umbral and self.polaridad == 1) or (x[self.caracteristica] < self.umbral and self.polaridad == -1):
                predicciones.append(1)
            else:
                predicciones.append(-1)

        return predicciones
        
        """
        #print("self.caracteristica = " + str(self.caracteristica))
        caracteristicas_seleccionadas = X[:, self.caracteristica]
        predicciones = np.where(self.polaridad * caracteristicas_seleccionadas > self.polaridad * self.umbral, 1, -1)
        
        return predicciones
                
        
        

##ADABOOST
class Adaboost:
    ## Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T
        self.A = A
        self.classifiers = []
        self.alphas = []

    
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose = False):
        # Obtener el número de observaciones y de características por observación de X
        n_observaciones, n_caracteristicas = X.shape
        # Iniciar pesos de las observaciones a 1/n_observaciones
        pesos = np.ones(n_observaciones) / n_observaciones
        # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        for t in range(self.T):
            mejorError = 1
            mejorClasificador = []
            mejorPrediccion = []
            clasificadoresT = []
            erroresT = []
            alphasT = []

            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
            for a in range(self.A):
                # Crear un nuevo clasificador débil aleatorio
                clasificadorDebil = DecisionStump(n_caracteristicas)
                clasificadoresT.append(clasificadorDebil)

                # Calcular predicciones de ese clasificador para todas las observaciones
                prediccion = clasificadorDebil.predict(X)

                # Calcular el error: comparar predicciones con los valores deseados y acumular los pesos de las observaciones mal clasificadas
                error = np.sum(pesos * (prediccion != Y))
                erroresT.append(error)

                # Actualizar mejor clasificador hasta el momento: el que tenga menor error
                if error < mejorError:
                    mejorClasificador = clasificadorDebil
                    mejorPrediccion = prediccion
                    mejorError = error
                
            # Calcular el valor de alfa y las predicciones del mejor clasificador débil
            alpha = 0.5 * np.log((1 - mejorError) / mejorError)
            alphasT.append(alpha)

            # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            pesos = pesos * np.exp(-alpha * Y * mejorClasificador.predict(X))

            # Normalizar a 1 los pesos
            pesos = pesos / np.sum(pesos)

            # Guardar el clasificador y el alpha en la lista de clasificadores de Adaboost
            self.classifiers.append(mejorClasificador)
            self.alphas.append(alpha)

            # Imprimir información de depuración si verbose es True
            if verbose: 
                mensaje = "Añadido clasificador {:>3}: {:>4}, {:>6.4f}, {:+2}, {:>8.6f}".format(t+1, mejorClasificador.caracteristica, mejorClasificador.umbral, mejorClasificador.polaridad, mejorError)
                print(mensaje)
    
    ## Método para obtener una predicción con el clasificador fuerte Adaboost
    def predict(self, X):
        # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        """
        predicciones = []
        for x in (X):
            predicciones.append(np.sign(np.sum([alpha * clasificador.predict(x) for alpha, clasificador in zip(self.alphas, self.classifiers)])))
        return predicciones
        """
        """
        aux = np.array([predicciones.predict(X) for predicciones, alfas in zip(self.classifiers, self.alphas)])
        predicciones = np.sign(np.sum(aux, axis=0))
        return predicciones
        """
        aux = np.array([alpha * clasificador.predict(X) for alpha, clasificador in zip(self.alphas, self.classifiers)])
        predicciones = np.sign(np.sum(aux, axis=0))
        return predicciones


def train_and_evaluate(clase, T, A, verbose=False):
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    """
    print(X_train.shape, X_train.dtype)
    print(Y_train.shape, Y_train.dtype)
    print(X_test.shape, X_test.dtype)
    print(Y_test.shape, Y_test.dtype)
    """
    #Filtra los datos de entrenamiento y test para quedarnos solo con la clase que queremos
    Y_train_clase = (Y_train == clase).astype(int)*2 - 1
    Y_test_clase = (Y_test == clase).astype(int)*2 - 1  

    #Crea un clasificador Adaboost con los parámetros T y A y lo entrena con los datos de entrenamiento
    adaboost = Adaboost(T, A)
    print("Entrenando clasificador Adaboost para el digito = " + str(clase) + " con T = " + str(T) + " y A = " + str(A) + "...")
    start = time.time()
    adaboost.fit(X_train, Y_train_clase, verbose)
    end = time.time()
    total_time = end - start
    print("Tiempo de entrenamiento: {:.2f} segundos".format(total_time))

    """
    decisionStump = DecisionStump(784)

    print(decisionStump.caracteristica)
    print(decisionStump.umbral)
    print(decisionStump.polaridad)

    predicciones = decisionStump.predict(X_train)
    #print(predicciones)
    """

    # Obtener las predicciones del clasificador Adaboost para los datos de entrenamiento y test
    y_train_pred = adaboost.predict(X_train)
    y_test_pred = adaboost.predict(X_test)

    # Calcular las tasas de acierto para los datos de entrenamiento y test
    y_train_accuracy = np.mean(y_train_pred == Y_train_clase)
    y_test_accuracy = np.mean(y_test_pred == Y_test_clase)

    # Imprimir las tasas de acierto
    print("Tasa de acierto (train, test) en datos de entrenamiento y tiempo: {:.2f}%, {:.2f}%, {:.3f} segundos".format(y_train_accuracy * 100, y_test_accuracy * 100, total_time))

#Main
if __name__ == "__main__":
    # Entrenar y evaluar el clasificador Adaboost para la clase 1 con T = 5 y A = 20
    train_and_evaluate(8, 20, 10, verbose=True)