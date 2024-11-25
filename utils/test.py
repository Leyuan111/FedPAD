import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load example data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a t-SNE model
model = TSNE(n_components=2, random_state=0)

# Fit and transform the data
transformed = model.fit_transform(X)

# Plotting
plt.figure(figsize=(8, 6))
for class_value in range(3):
    # Select points that belong to the current class
    ii = y == class_value
    plt.scatter(transformed[ii, 0], transformed[ii, 1], label=iris.target_names[class_value])

plt.legend()
plt.title('t-SNE visualization of Iris dataset')

# Save the plot
plt.savefig('t_sne_iris.png')
plt.show()