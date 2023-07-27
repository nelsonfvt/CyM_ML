import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('Log_Data.csv')

X_list = df['x'].values.tolist()
l_list = df['l'].values.tolist()

X = np.asarray(X_list)
l = np.asarray(l_list)

X = X.reshape((-1, 1))

clf = LogisticRegression(random_state=0).fit(X, l)

print('Ajuste del modelo')
print(clf.score(X, l))

# parametros del modelo
print(clf.coef_)
print(clf.intercept_)

print('Predicciones:')
# etiquetas predichas de las primeras 10 muestras
test = X[0:9, :]
pred = clf.predict(test)
print('Etiquetas:')
print(pred)

# probabilidad de pertenencia
prob_pred = clf.predict_proba(test)
print('Pertenencias a clases 0 - 1:')
print(prob_pred)