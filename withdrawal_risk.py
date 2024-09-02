# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 
from apprentices import labelled_df, unlabelled_df

y = labelled_df["withdrawal"].values
X = labelled_df[["last coaching session (days)", "manager survey (on track)"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

X_new = unlabelled_df[["last coaching session (days)", "manager survey (on track)"]].values

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 