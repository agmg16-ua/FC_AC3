from tablero import *
from variable import *
from dominio import *
from main import *

def forwardChecking(tablero, almacen):

    #Creación de las variables
    pos_inic = [-1, -1]
    pos_fin = [-1, -1]
    indice_var = 0
    var_horiz = []
    var_vert = []
    
    #Variables verticales
    for i in range (tablero.alto):
        for j in range (tablero.ancho):
            if tablero.getCelda(i, j) == '-':
                if pos_inic == [-1, -1]:
                    pos_inic = [i, j]

                pos_fin = [i, j]
                
                if j == tablero.ancho-1:
                    var_aux = Variable("H"+str(indice_var), pos_fin[1]-pos_inic[1]+1, pos_inic, pos_fin, almacen)
                    indice_var = indice_var + 1
                    var_horiz.append(var_aux)
                    pos_inic = [-1, -1]
                    pos_fin = [-1, -1]
                
            else:
                if pos_inic != [-1, -1]:
                    var_aux = Variable("H"+str(indice_var), pos_fin[1]-pos_inic[1]+1, pos_inic, pos_fin, almacen)
                    indice_var = indice_var + 1
                    var_horiz.append(var_aux)
                    pos_inic = [-1, -1]
                    pos_fin = [-1, -1]
    
    indice_var = 0
    #Variables horizontales
    for j in range (tablero.ancho):
        for i in range (tablero.alto):
            if tablero.getCelda(i, j) == '-':
                if pos_inic == [-1, -1]:
                    pos_inic = [i, j]

                pos_fin = [i, j]
                
                if i == tablero.alto-1:
                    var_aux = Variable("V"+str(indice_var), pos_fin[0]-pos_inic[0]+1, pos_inic, pos_fin, almacen)
                    indice_var = indice_var + 1
                    var_vert.append(var_aux)
                    pos_inic = [-1, -1]
                    pos_fin = [-1, -1]
                
            else:
                if pos_inic != [-1, -1]:
                    var_aux = Variable("V"+str(indice_var), pos_fin[0]-pos_inic[0]+1, pos_inic, pos_fin, almacen)
                    indice_var = indice_var + 1
                    var_vert.append(var_aux)
                    pos_inic = [-1, -1]
                    pos_fin = [-1, -1]

    for var in (var_horiz):
        print(var.nombre + ": " + str(var.tamanyo) + ", " + str(var.dominio.lista))

    for var in (var_vert):
        print(var.nombre + ": " + str(var.tamanyo) + ", " + str(var.dominio.lista))

                

