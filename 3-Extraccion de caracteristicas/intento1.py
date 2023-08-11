import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
iris = load_iris()
iris_data = iris.data
iris_labels = iris.target
feature_names = iris.feature_names

# Crear un DataFrame con los datos
import pandas as pd
iris_df = pd.DataFrame(iris_data, columns=feature_names)
iris_df['species'] = iris_labels

# Crear gráficos de dispersión
sns.pairplot(iris_df, hue='species', diag_kind='kde')
plt.show()

# Matriz de correlación
correlation_matrix = iris_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
species_names = iris.target_names

for species_idx, species_name in enumerate(species_names):
    species_data = iris_df[iris_df['species'] == species_idx]
    
    print(f"Statistics for {species_name}:\n")
    print(species_data.describe())
    print("\n")