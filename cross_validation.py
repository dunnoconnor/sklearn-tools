from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import GridSearchCV
import numpy as np
from apprentice_data import X_train, y_train

steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1,50)}

cv= GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
cv_pred = cv.predict
print(cv.best_score_)
print(cv.best_params_)
