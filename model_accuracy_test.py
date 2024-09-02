# Import the modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from apprentices import labeled_df
import numpy as np

# set withdrawal as target value
y = labeled_df["withdrawal"].values
# set features as last coaching session and manager survey
X = labeled_df[["days_since_workshop","days_since_coaching","manager_ni_rating","self_ni_rating","missing_assignments"]].values

# Split into training set (80% of data) and test set (20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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