from variable import *
from dominio import *

def main():
    v1 = Variable("nom", 4, [0,2], [0,5], "j")
    lista = []
    lista.append(v1)

    lista2 = []
    lista2.append(v1)

    lista[0].nombre = "pep"

    print(lista2[0].nombre)

if __name__=="__main__":
    main()