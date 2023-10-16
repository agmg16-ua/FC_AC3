#Alejandro Guillén Merino           48790456-G
from tablero import *
from variable import *
from dominio import *
from main import *

var_horiz = []
var_vert = []

cola_ac3 = []
cola_eliminados_ac3 = []

######################################################################### 
# Creación de variables
######################################################################### 
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


######################################################################### 
# Métodos de Forward Checking
######################################################################### 
def restaura_dominio(variable):
    for restriccion in variable.list_restr:
        restaurar = restriccion.dom_elim[len(restriccion.dom_elim)-1]
        for palabra in restaurar:
            restriccion.dominio.lista.append(palabra)
        
        restriccion.dom_elim.pop()


def ajusta_dominio(variable):
    indice = 0

    for restriccion in variable.list_restr:
        posicion = variable.pos_init[0] - restriccion.pos_init[0]
        podas = []

        for pal_dom in restriccion.dominio.lista:
            if pal_dom[posicion] != variable.palabra[indice]:
                podas.append(pal_dom)
            
        for eliminar in podas:
            restriccion.dominio.lista.pop(restriccion.dominio.lista.index(eliminar))
        
        indice = indice + 1

        restriccion.dom_elim.append(podas)
    
    for restriccion in variable.list_restr:
        if len(restriccion.dominio.lista) == 0:
            return False
    
    return True


def forward_checking(variable, indice_var):
    for i, pal_dom in enumerate(variable.dominio.lista):
        variable.palabra = pal_dom

        if ajusta_dominio(variable) == True:
            if indice_var != len(var_horiz) - 1:
                indice_var = indice_var + 1

                if forward_checking(var_horiz[indice_var], indice_var) == True:
                    return True
                
                restaura_dominio(variable)
                indice_var = indice_var - 1
            else:
                return True
        else:
            restaura_dominio(variable)
        
        if i  == len(variable.dominio.lista) - 1:
            return False


######################################################################### 
# Métodos de AC3
######################################################################### 
def imprimir_ac3():
    for variable in var_horiz:
        print("Nombre: " + variable.nombre + "; Posicion: [" + str(variable.pos_init[0]) + ", " + str(variable.pos_init[1]) + "]; Tipo: Horizontal; Dominio: [" + ", ".join(variable.dominio.lista) + ']')

    for variable in var_vert:
        print("Nombre: " + variable.nombre + "; Posicion: [" + str(variable.pos_init[0]) + ", " + str(variable.pos_init[1]) + "]; Tipo: Vertical; Dominio: [" + ", ".join(variable.dominio.lista) + ']')


def crear_cola_restricciones():
    for variableH in var_horiz:
        for restriccion in variableH.list_restr:
            cola_ac3.append([variableH, restriccion])
    
    for variableV in var_vert:
        for restriccion in variableV.list_restr:
            cola_ac3.append([variableV, restriccion])

def ac3():
    print("\n--DOMINIOS ANTES DE AC3")
    imprimir_ac3()

    crear_cola_restricciones()

    #for elemento in cola_ac3:
    #    print("[" + elemento[0].nombre + ", " + elemento[1].nombre + "]")

    #Aqui inicia AC3
    while cola_ac3:
        cambio = False
        variable_borrar = cola_ac3[0][0]
        variable_mirar = cola_ac3[0][1]

        #Posicion de interseccion de las palabras
        posicion_H = variable_borrar.list_restr.index(variable_mirar)
        posicion_V = variable_mirar.list_restr.index(variable_borrar)

        podar = []

        #Comprobar si cada palabra del dominio es valida
        for palabraH in variable_borrar.dominio.lista:
            existe_palabra = False
            for palabraV in variable_mirar.dominio.lista:
                if palabraH[posicion_H] == palabraV[posicion_V]:
                    existe_palabra = True
            
            if existe_palabra == False:
                cambio = True
                podar.append(palabraH)
        
        #Elminar palabras no validas
        for palabra in podar:
            variable_borrar.dominio.lista.remove(palabra)
        
        if len(variable_borrar.dominio.lista) == 0:
            return False
        
        if cambio == True:
            rescatar = []
            for pareja in cola_eliminados_ac3:
                if pareja[1] == variable_borrar:
                    rescatar.append(pareja)      

            for pareja in rescatar:
                cola_ac3.append(pareja)
                cola_eliminados_ac3.remove(pareja)

        cola_eliminados_ac3.append(cola_ac3[0])
        cola_ac3.remove(cola_ac3[0])  
    
    return True
    



######################################################################### 
# Método para llenar el tablero con las palabras
#########################################################################  
def llenar_tablero(tablero):
    for variable in var_horiz:
        posicion = variable.pos_init

        for i in range(variable.tamanyo):
            tablero.setCelda(posicion[0], posicion[1]+i, variable.palabra[i])


######################################################################### 
# Métodos de preparación
######################################################################### 
def preparando_forward(tablero, almacen, ac3):
    
    if ac3 == False:
        print("HOLA")
        var_horiz.clear()
        var_vert.clear()

        inicializar_variables(tablero, almacen)

        crear_restricciones()

    solucion = forward_checking(var_horiz[0], 0)

    if solucion == True:
        print("Se ha solucionado el crucigrama")
        llenar_tablero(tablero)
        return True
    else:
        print("No se ha solucionado el crucigrama")
        return False


def preparando_ac3(tablero, almacen):
    var_horiz.clear()
    var_vert.clear()
    cola_ac3.clear()
    cola_eliminados_ac3.clear()

    inicializar_variables(tablero, almacen)

    crear_restricciones()

    solucion = ac3()

    print("\n--DOMINIOS DESPUES DE AC3")
    imprimir_ac3()

    if solucion == True:
        print("AC3 ha finalizado de manera satisfactoria")
        return True
    else:
        print("AC3 no ha terminado bien")
        return False




