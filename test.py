import numpy
import math




def aprox(number, amount,max):
    posit = int(math.floor(((number/max)+1)*amount/2))
    return posit


nowa_pozycja_momentu = aprox(7.84, 5, 8)

print(nowa_pozycja_momentu)