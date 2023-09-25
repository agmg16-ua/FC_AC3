#Clase Variable
from dominio import *

class Variable:
    def __init__(self, nombre, tamanyo, pos_init, pos_fin, almacen):
        self.nombre = nombre
        self.tamanyo = tamanyo
        self.pos_init = pos_init
        self.pos_fin = pos_fin
        self.palabra = ""
        self.list_restr = []
        self.dominio = Dominio(tamanyo)

        dom = 0
        for dom_alm in almacen:
            if dom_alm.tam == tamanyo:
                self.dominio = dom_alm
                dom = 1
    


    