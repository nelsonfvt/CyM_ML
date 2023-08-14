# Code by Jonathan Guerrero

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from scipy.stats import norm

def plot_normal_distribution(data_array, label):
    # Create a histogram of the data
    sns.histplot(data_array, kde=True, label=f'{label} Data')

    # Fit a normal distribution to the data
    mu, std = norm.fit(data_array)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Fitted Normal Distribution: $\mu$ = {mu:.2f}, $\sigma$ = {std:.2f}')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Normal Distribution')
    plt.legend()

    plt.draw()  # Draw the plot
    plt.pause(0.001)  # Pause to allow interaction

# import some data to play with
iris = datasets.load_iris()

data = iris.data # Take the data
target = iris.target # Take the target

# Separe the data per target
data_per_target = {}
for i in range(len(target)):
    if target[i] not in data_per_target:
        data_per_target[target[i]] = [data[i]]
    else:
        data_per_target[target[i]].append(data[i])

# Convert the lists to numpy arrays
for key in data_per_target:
    data_per_target[key] = np.array(data_per_target[key])

# print(data_per_target) # Uncomment this line if you want print the data at the terminal

# Select the variable to graph (The irirs dataset have four variables, so with this number [0-3] you can select a single one)
var = 0

# Set non-blocking mode
plt.ion()

# Plot a normal distribution
for target_label, data_array in data_per_target.items():
    plot_normal_distribution(data_array[:, var], f'Target {target_label}')

plt.ioff()  # Turn off non-blocking mode after all plots are displayed
plt.show()  # Keep the final plot displayed until manually closed