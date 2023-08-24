import numpy as np
from sklearn.linear_model import orthogonal_mp, OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib.image import imread


def OMP(D, y, noncero_coef=None, tol=None):
    return orthogonal_mp(D, y, n_nonzero_coefs=noncero_coef, tol=tol, precompute=True )

def K_SVD(X, D, noncero_coef, tol ,iter):
    f, K = D.shape
    for n in range(iter):
        print('iteracion K-SVD: ' + str(n))
        W = OMP(D, X, tol=tol) #noncero_coef=noncero_coef
        R = X - D.dot(W)
        for k in range(K):
            wk = np.nonzero(W[k,:])[0] #indices no cero
            if len(wk) == 0:
                continue
            DW = np.empty([9, 0])
            for j in range(len(wk)):
                DW = np.c_[DW, D[:,k]*W[k,wk[j]]]
            Ri = R[:, wk] + DW
            U, s, Vh = np.linalg.svd(Ri)
            D[:, k] = U[:, 0]
            W[k, wk] = s[0] * Vh[0, :]
            DW = np.empty([9, 0])
            for j in range(len(wk)):
                DW = np.c_[DW, D[:,k]*W[k,wk[j]]]
            R[:, wk] = Ri - DW
    
    return D

prec = 0.01

# Cargando imagenes
lenna = imread('lenna.jpg')
barbe = imread('barbara.jpg')

# tamaño atomo 3x3 = 9
patch_size = 3
# tamaños imagenes
s_lenna = lenna.shape
s_barbe = barbe.shape

# Construyendo Diccionario Do
Do = np.empty([9, 0]) #vacio

# Extrayendo parches de Lenna
for i in range(2, s_lenna[0], patch_size): #filas
    for j in range(2,s_lenna[1], patch_size): #columnas
        patch = lenna[i-1:i+2, j-1:j+2]
        patch = np.reshape(patch,(9, 1))
        Do = np.c_[Do, patch]

print('Tamaño del diccionario original:')
print(Do.shape)
#Eliminar linear dependencies del diccionario
# se usa un algoritmo de agrupamiento
k = round(Do.shape[1]/10) #Decima parte del original
kmeans = KMeans(k,max_iter=200,n_init='auto', random_state=0, init='k-means++').fit(Do.T)
D1 = kmeans.cluster_centers_.T
print('Tamaño diccionario reducido:')
print(D1.shape)

# Extrayendo algunos parches de Barbara 
X = np.empty([9, 0]) #vacio
fila = round(s_barbe[1]/2)
for i in range(2, s_barbe[1]-1, patch_size): #una fila
    patch = barbe[fila-1:fila +2, i-1:i+2]
    patch = np.reshape(patch, (9, 1))
    X = np.c_[X, patch]

print('Tamaño de X:')
print(X.shape)

# OMP test
# print('Prueba de Ortogonal Matching Pursuit:')
# W = OMP(Do, X[:, 0:3],tol=prec)
# F = Do.dot(W)
# print('Matriz Reconstruida:')
# print(F)
# print('Matriz Original:')
# print(X[:, 0:3])
# print('MSE')
# print(mean_squared_error(X[:, 0:3], F))

# K-SVD test
D_1 = K_SVD(X, D1, noncero_coef=15,tol=prec, iter=10)
print('Tamaño Diccionario K-SVD:')
print(D_1.shape)

# Extrayendo algunos parches de Barbara 
X_t = np.empty([9, 0]) #vacio
fila = round(s_barbe[1]/2)+2
for i in range(2, s_barbe[1]-1, patch_size): #una fila
    patch = barbe[fila-1:fila +2, i-1:i+2]
    patch = np.reshape(patch, (9, 1))
    X_t = np.c_[X_t, patch]

W = OMP(D_1, X_t[:, 0:3],tol=prec)
F = D_1.dot(W)
print('Matriz Reconstruida:')
print(F)
print('Matriz Original:')
print(X_t[:, 0:3])
print('MSE')
print(mean_squared_error(X_t[:, 0:3], F))
