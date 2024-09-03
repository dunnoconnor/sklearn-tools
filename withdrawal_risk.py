# Import KNeighborsClassifier and labeled / unlabeled dataframes
from sklearn.neighbors import KNeighborsClassifier 
from apprentice_data import labeled_df, unlabeled_df, print_at_risk

# set withdrawal as target value
y = labeled_df["withdrawal"].values
# set features as last coaching session, manager survey, etc
X = labeled_df[["days_since_workshop","days_since_coaching","manager_ni_rating","self_ni_rating","missing_assignments"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# store values of unlabeled data
X_new = unlabeled_df[["days_since_workshop","days_since_coaching","manager_ni_rating","self_ni_rating","missing_assignments"]].values

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 

# Print apprentices predicted to withdraw
print_at_risk(y_pred)
