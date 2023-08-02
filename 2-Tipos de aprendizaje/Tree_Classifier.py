from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

r = export_text(clf, feature_names=iris['feature_names'])
print(r)

y_pred = clf.predict(X_test)

print("Número de puntos para test:")
N = X_test.shape[0]
print(N)
print("Número de errores:")
f = (y_test != y_pred).sum()
print(f)

print('Porcentaje de error:')
print(f/N*100)