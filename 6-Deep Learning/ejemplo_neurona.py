import numpy as np

def neurona(inp, w, b):
    f = np.matmul(inp,w) + b
    return 1/(1 + np.exp(-f)) #sigmoide

def EQM(err):
    return np.sum( np.square(err)) * 0.5

def Err(out, yc):
    return out - yc

def delta_rule(inp,out,w,b):
    alpha = 0.1
    
    Epoch = 1
    E = 1000

    while E > 0.01 and Epoch < 5000:

        yc = neurona(inp,w,b)
    
        err = Err(out, yc)
        E = EQM(err)

        print('Epoca: ' + str(Epoch))
        print('Error: ' + str(E))

        df = yc * (1-yc)

        for k in range(0,len(yc)):
            
            #df = yc[k] * (1 - yc[k]) # sigmoide derivada
            delta = alpha * err[k] * df[k]
            delt_w = delta * inp[k,:]
            b = b + delta
            w = w + delt_w

        yc = neurona(inp,w,b)
        err = Err(out, yc)
        E = EQM(err)
        Epoch = Epoch+1
        

    return w,b

input = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
output = np.array([0.0, 1.0, 1.0, 1.0])

w = np.array([0.1, 0.1])
b = -0.1

w_f,b_f = delta_rule(input, output, w, b)
print('pesos finales:')
print(w_f)
print('bias final:')
print(b_f)
y_f = neurona(input,w_f, b_f)
print('salidas resultantes:')
print(y_f)