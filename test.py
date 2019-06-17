import numpy
import math




def aprox(number, amount,max):
    posit = int(math.floor(((number/max)+1)*amount/2))
    return posit


nowa_pozycja_momentu = aprox(7.84, 5, 8)

def oblicz_theta (y,x):
    if (x>0):
        th = math.acos(y) - math.pi
    else:
        th = math.pi - math.acos(y)
    th += 0.130899
    if th >= math.pi:
        th = th - 2*math.pi
    return th

def oblicz_thetar (y,x):
    if (x>0):
        return math.acos(y) - math.pi
    else:
        return math.pi - math.acos(y)

print(oblicz_theta(1,0))
print(aprox(oblicz_theta(1,0),24,math.pi))




print(oblicz_thetar(1, 0))
print(aprox(oblicz_thetar(1, 0), 24, math.pi))

print(1+2*3)