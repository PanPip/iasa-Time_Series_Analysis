import numpy as np
from random import gauss
import math

def row_multiply_line(x1, x2, dim):
    A = [[0] * dim for i in range(dim)]
    for i in range(dim):
        for j in range(dim): 
            A[i][j] = x1[i] * x2[j]
    return np.array(A)

n = 100

file = open("mytest.txt", 'r');
p = int(file.readline())
q = int(file.readline())

coefs = [0] * (p + q + 2)
for i in range(p + q + 2):
    coefs[i] = float(file.readline())   

v = [0] * (n + 1)
for i in range(1, n + 1):
    v[i] = gauss(0, 1)
    
def create_y(coefs):
    y = [0] * (n + 1)
    for i in range(1, max(p, q) + 1):
        y[i] = v[i]
    for i in range(max(p, q) + 1, n + 1):
        temp = []
        for j in range(1, p + 1):
            temp.append(y[i - j])
        for j in range(q + 1):
            temp.append(v[i - j])
        coefs = np.array(coefs)
        temp = np.array(temp)
        y[i] = coefs[0] + coefs[1:] @ temp
    return np.array(y)                

y = create_y(coefs)

# перебираем модели от АРКС(1, 1) до АРКС(3, 3)
for p in range(1, 4):
    for q in range(1, 4):
        print("\np = ", p, "; q = ", q)

        dimi_X = n - max(p, q)    # dimension i of matrix X
        dimj_X = 2 + p + q    #dimension j of matrix X  
        #X = np.array([[0] * dimj_X for i in range(dimi_X)])
        X = [[0] * dimj_X for i in range(dimi_X)]
        for i in range(dimi_X):
            help = []
            help.append(1)
            for j in range(1, p + 1):
                help.append(y[(i + 1) + p - j])
            for j in range(q + 1):
                help.append(v[(i + 1) + q - j])
            help = np.array(help)
            for j in range(dimj_X):
                X[i][j] = help[j]
        X = np.array(X)
        theta = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y[max(p, q) + 1:]
        print(theta)

        S = np.linalg.norm(y[max(p, q) + 1:] - X @ theta) ** 2
        print("S = ", S)
        
        _y = create_y(theta)
        R2 = np.var(_y) / np.var(y)
        print("R2 = ", R2)
        
        IKA = n * math.log(S) + 2 * (p + q + 1)
        print("IKA = ", IKA)

        alpha = np.array([0] * dimj_X)
        beta = 100

        P = np.array([[0] * dimj_X for i in range(dimj_X)])
        for i in range(dimj_X):
            for j in range(dimj_X):
                if i == j :
                    P[i, j] = 1
        P = beta * P
        theta = alpha
        for i in range(1, dimi_X):
            P = P - (1 / (1 + (P @ X[i - 1]) @ X[i - 1])) * (P @ row_multiply_line(X[i - 1], X[i - 1], dimj_X) @ P)
            theta = theta + (y[i + max(p, q)] - X[i - 1] @ theta) * P @ X[i - 1]
        print(theta)

        S = np.linalg.norm(y[max(p, q) + 1:] - X @ theta) ** 2
        print("S = ", S)
        
        _y = create_y(theta)
        R2 = np.var(_y) / np.var(y)
        print("R2 = ", R2)
        
        IKA = n * math.log(S) + 2 * (p + q + 1)
        print("IKA = ", IKA)
    
    






        
        
        
        
        
