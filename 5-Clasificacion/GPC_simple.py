import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, log_loss

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, 0:2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)

kernel = 1.0 * RBF([1.0])

classifier_x = GaussianProcessClassifier(kernel=kernel, optimizer=None).fit(X,y)
classifier_t = GaussianProcessClassifier(kernel=kernel).fit(X,y)

# create a mesh to plot in
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Probabilidades antes de entrenamiento

plt.subplot(1, 2, 1)
Z = classifier_x.predict_proba(np.c_[xx.ravel(), yy.ravel()])

titles = ["Sin entrenar", "Entrenado"]
# Put the result into a color plot
Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(
    "%s" % (titles[0])
)

# Probabilidades despues de entrenamiento

plt.subplot(1,2, 2)
Zt = classifier_t.predict_proba(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Zt = Zt.reshape((xx.shape[0], xx.shape[1], 3))
plt.imshow(Zt, extent=(x_min, x_max, y_min, y_max), origin="lower")
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(
    "%s" % (titles[1])
)

print(
    "Log marginal Likelihood (sin entrenar): %.3f"
    % classifier_x.log_marginal_likelihood(classifier_x.kernel_.theta)
)
print(
    "Log marginal Likelihood (entrenado): %.3f"
    % classifier_t.log_marginal_likelihood(classifier_t.kernel_.theta)
)

print(
    "Accuracy: %.3f (sin entrenar) %.3f (entrenado)"
    % (
        accuracy_score(y, classifier_x.predict(X)),
        accuracy_score(y, classifier_t.predict(X))
    )
)

print(
    "Log-loss: %.3f (sin entrenar) %.3f (entrenado)"
    % (
        log_loss(y, classifier_x.predict_proba(X)),
        log_loss(y,classifier_t.predict_proba(X))
        )
)

plt.tight_layout()
plt.show()