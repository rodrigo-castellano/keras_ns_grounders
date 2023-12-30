import numpy as np
from itertools import combinations

def pol_per_grade(x,grado):
    polynomial = 0
    for tuple in combinations(x, grado):
        monomial = np.product(tuple)
        polynomial += monomial
    return polynomial

def fuzzy_xor(x_array):
    l = len(x_array)
    sum = 0
    for g in range(l):
        coeff = ((-1)**(g))*(g+1)
        poly_per_grade = coeff*pol_per_grade(x,g+1)
        sum += poly_per_grade
    return sum



x=np.array([0,0.1,0,0.8,0,0.1,1])
# print("res",pol_per_grade(x,2))
print("result xor",fuzzy_xor(x))