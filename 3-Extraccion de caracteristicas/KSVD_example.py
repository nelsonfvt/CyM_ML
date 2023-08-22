import numpy as np
from sklearn.linear_model import orthogonal_mp

def OMP(D, Y, noncero_coef=None, tol=None):
    return orthogonal_mp(D, Y, noncero_coef, tol, precompute=True)

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
            

