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
    
    #Variables horizontales
    for i in range (tablero.alto):
        for j in range (tablero.ancho):
            if tablero.getCelda(i, j) == '-':
                if pos_inic == [-1, -1]:
                    pos_inic = [i, j]

                pos_fin = [i, j]
                
                if j == tablero.ancho-1:
                    var_aux = Variable("H"+str(indice_var), pos_fin[1]-pos_inic[1]+1, pos_inic, pos_fin, almacen)
                    #crear_dominio(var_aux)
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
    #Variables verticales
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
                if hor.pos_init[1]+i == ver.pos_init[1]:
                    if hor.pos_init[0] >= ver.pos_init[0] and hor.pos_init[0] <= ver.pos_fin[0]:
                        hor.list_restr.append(ver)
                        ver.list_restr.append(hor)

def restaura_dominio(variable):
    for restriccion in variable.list_restr:
        restaurar = restriccion.dom_elim[len(restriccion.dom_elim)-1]
        for palabra in restaurar:
            restriccion.dominio.lista.append(palabra)
        
        restriccion.dom_elim.pop()

def ajusta_dominio(variable):
    indice = 0

    for restriccion in variable.list_restr:
        #print(restriccion.nombre)
        posicion = variable.pos_init[0] - restriccion.pos_init[0]
        podas = []

        for pal_dom in restriccion.dominio.lista:
            #print(pal_dom)
            #print(pal_dom[posicion] + ' ' + variable.palabra[indice])
            if pal_dom[posicion] != variable.palabra[indice]:
                #print("ENTRAAAAAAAAAAAAAAAAA")
                podas.append(pal_dom)
            
        for eliminar in podas:
            restriccion.dominio.lista.pop(restriccion.dominio.lista.index(eliminar))
        
        indice = indice + 1

        restriccion.dom_elim.append(podas)
        #print(" ")
        #print(" ")
    
    for restriccion in variable.list_restr:
        if len(restriccion.dominio.lista) == 0:
            return False
    
    return True

def forward_checking(variable, indice_var):
    for i, pal_dom in enumerate(variable.dominio.lista):
        #print(variable.nombre + ' ' + str(indice_var))
        variable.palabra = pal_dom

        if ajusta_dominio(variable) == True:
            #print("holaa")
            if indice_var != len(var_horiz) - 1:
                indice_var = indice_var + 1

                if forward_checking(var_horiz[indice_var], indice_var) == True:
                    return True
                
                else:
                    restaura_dominio(variable)
                    indice_var = indice_var - 1
            else:
                return True
        else:
            restaura_dominio(variable)
        
        if i  == len(variable.dominio.lista) - 1:
            return False
        
def llenar_tablero(tablero):
    for variable in var_horiz:
        posicion = variable.pos_init

        for i in range(variable.tamanyo):
            tablero.setCelda(posicion[0], posicion[1]+i, variable.palabra[i])

def preparando(tablero, almacen):
    var_horiz.clear()
    var_vert.clear()

    inicializar_variables(tablero, almacen)

    crear_restricciones()

    solucion = forward_checking(var_horiz[0], 0)

    if solucion == True:
        print("Se ha solucionado el crucigramaaaaa")
        llenar_tablero(tablero)
        return True
    else:
        print("No se ha solucionado el crucigramaaaaa")
        return False




