import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('MLin_Data.csv')

y_list = df['y'].values.tolist()
df.head()
x_matr = df[df.columns[1:6]]

y = np.asarray(y_list)
X = x_matr.to_numpy()

reg = LinearRegression().fit(X,y)
print('Ajuste del modelo:')
print(reg.score(X,y))

print('Parámetros encontrados:')
print(reg.coef_)
print(reg.intercept_)

test = np.array([15, -1, -3, 23, 7])
pred = reg.predict(test.reshape(1, -1))
print('Prediccion')
print(pred)