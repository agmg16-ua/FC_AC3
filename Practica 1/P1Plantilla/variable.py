#Clase Variable
from dominio import *

class Variable:
    def __init__(self, nombre, tamanyo, pos_init, pos_fin, orientacion):
        self.nombre = nombre
        self.tamanyo = tamanyo
        self.pos_init = pos_init
        self.pos_fin = pos_fin
        self.palabra = ""
        self.orientacion = orientacion #V (vertical) o H (horizontal)
        self.list_restr = []
        self.dominio = Dominio(tamanyo)
    
    def getDom(self):
        self.dominio.getLista
    
    def getPal(self):
        return self.palabra

    