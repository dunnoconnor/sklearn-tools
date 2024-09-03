from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier 
from apprentice_data import X_train, X_test, y_train, y_test, print_at_risk

steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier(n_neighbors=13))]
pipeline = Pipeline(steps)

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test,y_test))

# Print the predictions
print("Predictions: {}".format(y_pred)) 

# Print each apprentice who is predicted to withdraw
print_at_risk(y_pred)