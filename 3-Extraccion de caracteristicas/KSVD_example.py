import numpy as np
from sklearn.linear_model import orthogonal_mp, OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler
from matplotlib.image import imread


def OMP(D, y, noncero_coef=None, tol=None):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=noncero_coef, tol=tol)
    #scaler = StandardScaler()
    #D = scaler.fit_transform(D)
    omp.fit(D, y)
    return omp.coef_
    #return orthogonal_mp(D, Y, n_nonzero_coefs=noncero_coef, tol=tol, precompute=True )

def K_SVD(X, D, K, noncero_coef ,iter):

    for n in range(iter):
        print(n)
        W = OMP(D, X, noncero_coef)
        R = X - D.dot(W)
        for k in range(K):
            wk = np.nonzero(W[k,:])[0]
            Ri = R[:, wk] + D[:,k].dot(W[None,k,wk])
            U, s, Vh = np.linalg.svd(Ri)
            D[:, k] = U[:, 0]
            W[k, wk] = s[0] * Vh[0, :]
            R[:, wk] = Ri - D[:, k, None]. dot(W[None,k,wk])
    
    return D
            
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
        patch = patch / np.linalg.norm(patch)
        Do = np.c_[Do, patch] #append(Do, patch, axis=1)

# Extrayendo algunos parches de Barbara 
X = np.empty([9, 0]) #vacio
fila = round(s_barbe[1]/2)
for i in range(2, s_barbe[1]-1, patch_size): #una fila
    patch = barbe[fila-1:fila +2, i-1:i+2]
    patch = np.reshape(patch, (9, 1))
    patch = patch / np.linalg.norm(patch)
    X = np.c_[X, patch]

print('Tamaño del diccionario:')
print(Do.shape)
print('Tamaño de X:')
print(X.shape)

W = OMP(Do, X[:, 0:1])