import numpy as np
from sklearn.decomposition import NMF

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
print("Dataset ejemplo:")
print(X)

model = NMF(n_components=2, init='random', random_state=0)

W = model.fit_transform(X)
print("Descomposición")
print(W)

H = model.components_
print("Componentes")
print(H)