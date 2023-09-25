from tablero import *
from variable import *
from dominio import *
from main import *

var_horiz = []
var_vert = []

def inicializar_variables(tablero, almacen):
        #Creación de las variables
    pos_inic = [-1, -1]
    pos_fin = [-1, -1]
    indice_var = 0
    
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
    

def crear_restricciones():

    #se añade a la lista de restricciones la variable que intersecte
    for hor in var_horiz:
        for ver in var_vert:
            for i in range(hor.pos_fin[1]-hor.pos_init[1]+1):
                print("hola1: " + hor.nombre + ' ' + str(hor.pos_init) + ' ' + str(hor.pos_fin))
                print("hola2: " + ver.nombre + ' ' + str(ver.pos_init) + ' ' + str(ver.pos_fin))
                if hor.pos_init[1]+i == ver.pos_init[1]:
                    if hor.pos_init[0] >= ver.pos_init[0] and hor.pos_init[0] <= ver.pos_fin[0]:
                        hor.list_restr.append(ver.nombre)
                        ver.list_restr.append(hor.nombre)
                        print("yes")
                
                print()


def forwardChecking(tablero, almacen):
    var_horiz.clear()
    var_vert.clear()

    inicializar_variables(tablero, almacen)

    crear_restricciones()

    for var in (var_horiz):
        print(var.nombre + ": " + str(var.tamanyo) + ", " + str(var.list_restr))

    for var in (var_vert):
        print(var.nombre + ": " + str(var.tamanyo) + ", " + str(var.list_restr))




