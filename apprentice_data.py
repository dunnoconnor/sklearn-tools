# import pandas for dataframes
import pandas as pd
from sklearn.model_selection import train_test_split

# store csv data in dataframes
labeled_df = pd.read_csv('labeled_data3.csv')
unlabeled_df = pd.read_csv('unlabeled_data3.csv')

# set withdrawal as target value
y = labeled_df["withdrawal"].values
# set features as last coaching session and manager survey
X = labeled_df[["days_since_workshop","days_since_coaching","manager_ni_rating","self_ni_rating","missing_assignments"]].values

# Split into training set (80% of data) and test set (20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print each apprentice predicted to withdraw
def print_at_risk(pred):
    apps = unlabeled_df.to_numpy()
    count = 0
    for i in range(len(pred)):
        if pred[i] == True:
            count += 1
            print(apps[i])
# Print the count of apprentices predicted to withdraw
    print(count)