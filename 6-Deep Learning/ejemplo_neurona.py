import numpy as np

def neurona(inp, w):
    f = np.matmul(inp,w)
    return 1/(1 + np.exp(-f)) #sigmoide

def EQM(err):
    return np.sum( np.square(err)) * 0.5

def Err(out, yc):
    return out - yc

def delta_rule(inp,out,w):
    alpha = 0.1
    yc = neurona(inp,w)
    
    err = Err(out, yc)
    E = EQM(err)
    Epoch = 1

    while E > 0.01 and Epoch < 1500:

        print('Epoca: ' + str(Epoch))
        print('Error: ' + str(E))     
        for k in range(0,len(yc)):
            
            df = yc[k] * (1 - yc[k]) # sigmoide derivada
            delt_w = alpha * err[k] * df * inp[k,:]
            w = w + delt_w

        yc = neurona(inp,w)
        err = Err(out, yc)
        E = EQM(err)
        Epoch = Epoch+1
        

    return w

input = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
output = np.array([0.0, 1.0, 1.0, 1.0])

w = np.array([0.5, 0.5])

w_f = delta_rule(input, output, w)
print(w_f)
y_f = neurona(input,w_f)
print(y_f)