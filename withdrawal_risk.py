# Import KNeighborsClassifier and labeled / unlabeled dataframes
from sklearn.neighbors import KNeighborsClassifier 
from apprentices import labeled_df, unlabeled_df

# set withdrawal as target value
y = labeled_df["withdrawal"].values
# set features as last coaching session, manager survey, etc
X = labeled_df[["days_since_workshop","days_since_coaching",'positive_manager_survey',"positive_self_survey","missing_assignments"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# store values of unlabeled data
X_new = unlabeled_df[["days_since_workshop","days_since_coaching",'positive_manager_survey',"positive_self_survey","missing_assignments"]].values

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 

# Print each apprentice who is predicted to withdraw
apps = unlabeled_df.to_numpy()
count = 0
for i in range(len(y_pred)):
    if y_pred[i] == True:
        count += 1
        print(apps[i])

# Print the count of apprentices predicted to withdraw
print(count)
