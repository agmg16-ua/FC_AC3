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
                if hor.pos_init[1]+i == ver.pos_init[1]:
                    if hor.pos_init[0] >= ver.pos_init[0] and hor.pos_init[0] <= ver.pos_fin[0]:
                        hor.list_restr.append(ver)
                        ver.list_restr.append(hor)


def forward_checking(variable, indice_var):
    for pal_dom in variable.dominio.lista:
        variable.nombre = pal_dom
        letra = 0
        for pal_restr in variable.list_restr:
            indice = var_vert.index(pal_restr)
            var_vert[indice]
            posicion_letra = variable.pos_init[0] - pal_restr.pos_init[0]

            for pal_restr_dom in var_vert[indice].dominio.lista:
                if pal_restr_dom[posicion_letra] != variable.nombre[0]:
                    var_vert[indice].dominio.lista.remove(pal_restr_dom)
                
            if len(var_vert[indice].dominio.lista) == 0:
                print("Se ha vaciado un dominio")
                return 1

            letra = letra + 1
        
        if len(var_horiz) == indice_var:
            return 0
        
        seguir = forward_checking(var_horiz[indice_var + 1], indice_var + 1)

        if seguir == 0:
            return 0           
            
        

def preparando(tablero, almacen):
    var_horiz.clear()
    var_vert.clear()

    inicializar_variables(tablero, almacen)

    crear_restricciones()

    solucion = forward_checking(var_horiz[0], 0)

    if solucion == 0:
        print("Se ha solucionado el crucigrama")
    else:
        print("No se ha solucionado el crucigrama")




