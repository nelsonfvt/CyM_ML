import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('Lin_Data.csv')

X_list = df['x'].values.tolist()
y_list = df['y'].values.tolist()

#print(type(X[0]))

X = np.asarray(X_list)
y = np.asarray(y_list)

X = X.reshape((-1, 1))

reg = LinearRegression().fit(X,y)
print('Ajuste del modelo:')
print(reg.score(X,y))

print('Parámetros encontrados:')
print(reg.coef_)
print(reg.intercept_)

test = np.arange(14,30)
pred = reg.predict(test.reshape(-1, 1))
print('Predicciones')
print(pred)