# Import the modules
from sklearn.neighbors import KNeighborsClassifier 
from apprentice_data import X_train, y_train, X_test, y_test
import numpy as np

# Create neighbors
neighbors = np.arange(1,13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fit the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)

# Print the accuracy for each k of neighbors
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)