# Import KNeighborsClassifier and labelled / unlabelled dataframes
from sklearn.neighbors import KNeighborsClassifier 
from apprentices import labelled_df, unlabelled_df

# set withdrawal as target value
y = labelled_df["withdrawal"].values
# set features as last coaching session, manager survey, etc
X = labelled_df[["days_since_workshop","days_since_coaching",'positive_manager_survey',"positive_self_survey","missing_assignments"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# store values of unlabelled data
X_new = unlabelled_df[["days_since_workshop","days_since_coaching",'positive_manager_survey',"positive_self_survey","missing_assignments"]].values

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 

# Print each apprentice who is predicted to withdraw
apps = unlabelled_df.to_numpy()
for i in range(len(y_pred)):
    if y_pred[i] == True:
        print(apps[i])
    
