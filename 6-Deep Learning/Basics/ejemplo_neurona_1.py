import numpy as np


def neurona(inp, w, b):
    f = np.matmul(inp,w) + b
    #return np.tanh(f) #Tang hiperbolica
    return 1/(1 + np.exp(-f)) #sigmoide

input = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
output = np.array([0.0, 1.0, 1.0, 1.0])

#Ajuste manualmente
w = np.array([0.1, 0.1])
b = -0.1

y = neurona(input,w,b)
print('salidas obtenidas:')
print(y)

err = output - y
print('error:')
print(err)