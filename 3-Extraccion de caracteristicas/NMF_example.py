import numpy as np
from sklearn.decomposition import NMF

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
print("Dataset ejemplo (X):")
print(X)

model = NMF(n_components=2, init='random', random_state=0)

W = model.fit_transform(X)
print("Datos transformados W:")
print(W)

H = model.components_
print("Componentes H:")
print(H)

Xr = W.dot(H)

print("X reconstruida (W*H):")
print(Xr)

E = X-Xr
print("Error de reconstrucción (X-Xr):")
print(E)