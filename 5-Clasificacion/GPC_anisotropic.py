# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, 0:2]  # we only take the first two features for visualization
y = iris.target

n_features = X.shape[1]

kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

classifier = GaussianProcessClassifier(kernel=kernel).fit(X, y)

y_pred = classifier.predict(X)

accuracy = accuracy_score(y, y_pred)
print("Accuracy (train) for anisotropic GPC: %0.1f%% " % (accuracy * 100))

plt.figure(figsize=(3 * 2, 2))
plt.subplots_adjust(bottom=0.2, top=0.95)

xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]

probas = classifier.predict_proba(Xfull)
n_classes = np.unique(y_pred).size

for k in range(n_classes):
    plt.subplot(1, n_classes, k + 1)
    plt.title("Class %d" % k)
    if k == 0:
        plt.ylabel('Anisotropic GPC')
    imshow_handle = plt.imshow(
        probas[:, k].reshape((100, 100)), extent=(3, 9, 1, 5), origin="lower"
    )
    plt.xticks(())
    plt.yticks(())
    idx = y_pred == k
    if idx.any():
        plt.scatter(X[idx, 0], X[idx, 1], marker="o", c="w", edgecolor="k")

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")

plt.show()