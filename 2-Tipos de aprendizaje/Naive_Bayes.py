from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)

# Imprimiendo dataset
print('Datos de entrada:')
print(X)

print('Salidas/etiquetas:')
print(y)

#Separando el conjunto en train y test 50 - 50
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# generando modelo
gnb = GaussianNB()

# entrenando clasificador
gnb.fit(X_train, y_train)

# Prediccion con test
y_pred = gnb.predict(X_test)

print("Número de puntos para test:")
N = X_test.shape[0]
print(N)
print("Número de errores:")
f = (y_test != y_pred).sum()
print(f)

print('Porcentaje de error:')
print(f/N*100)