import numpy as np


def neurona(inp, w):
    f = np.matmul(inp,w)
    #return np.tanh(f) #Tang hiperbolica
    return 1/(1 + np.exp(-f)) #sigmoide

input = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
output = np.array([0.0, 1.0, 1.0, 1.0])

w = np.array([0.5, 0.5]) #Ajuste manualmente

y = neurona(input,w)
print('salidas obtenidas:')
print(y)

err = output - y
print('error:')
print(err)