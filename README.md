# Predictive Model - Apprentice Withdrawal Risk 

## Data Privacy Disclaimer
All data used in this proof of concept is generated using [mockaroo](https://www.mockaroo.com/). Data features (including first and last name) were mocked for testing purposes and are not representative of actual people or programs.

## Use Case
This is a proof of concept for using [scikit-learn](https://scikit-learn.org/stable/index.html) to create a predictive model capable of analyzing if an apprentice is likely to withdraw from an apprenticeship program. The end user of this tool would be an apprenticeship provider looking to make more targeted interventions to improve apprentice retention and program completion.

## Data Sets
Two data sets are included in this repository. The first is [labeled data](/labeled_data3.csv) which contains 1000 profiles of apprentices who have completed the program. The features of this data set are:
* Days since the apprentice last attended a workshop (integer)
* Days since the apprentice last attended a coaching session (integer)
* Whether the manager evalutation rated the apprentice 'needs improvement' (boolean)
* Whether the apprentice rated 'needs improvement' in their self-evaluation (boolean)
* Total number of assignments the apprentice is missing (integer)
* Whether the apprentice withdrew before completing the program (boolean)

The second data set is [unlabeled data](/unlabeled_data3.csv) which contains 1000 profiles of apprentices currently on program. This data has all the features of the labeled data **except** for withdrawal status.

## Machine Learning
This predictive model is based on scikit-learn: a library of open-source tools for predictive data analysis. This specific project is a supervised learning model using k-neighbors classification.

[Supervised Learning](https://www.ibm.com/topics/supervised-learning)

> Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled data sets to train algorithms that to classify data or predict outcomes accurately.

[Nearest Neighbors Classification](https://scikit-learn.org/stable/modules/neighbors.html#classification)

> Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

> The k-neighbors classification in KNeighborsClassifier is the most commonly used technique. The optimal choice of the value is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct.

**The target value the model predicts is a boolean: whether a given apprentice will (or will not) withdraw before completing this program.**

All the other non-identifying features of the data are used to find relationships. As withdrawal status is a discrete value (rather than continuous), this is a classification model not a regression model.

## Approach
I completed the following steps to train and implement the model:

Convert the CSV data to [pandas DataFrames](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe) in the file [apprentice_data.py](/apprentice_data.py).

Import [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#kneighborsclassifier) from scikit-learn and the labeled and unlabeled DataFrames to [withdrawal_risk.py](/withdrawal_risk.py).

Set the X and y values of the model. Withdrawal status is the y (target) value and the other features are X values. Identifying features like first and last name are excluded from the model.

Create a k-neighbors classifier with an arbitrary number of neighbors to revise later based on accuracy testing. Fit the classifier to the data using the fit() method. Predict the labels for the unlabeled data using the predict() method. For improved readability, separately print each apprentice who is predicted to withdraw along with a count of at-risk apprentices.

The result of running this model was a list of 62 out of 1000 apprentices predicted to withdraw before completing the program.

## Accuracy Testing
After generating a set of predictions, my next step was to test the model using a range of k-neighbor values. A larger k value suppresses the effects of noise, but makes the classification boundaries less distinct. By testing the accuracy of the model at multiple k values, I am able to determine the optimal k value to avoid underfitting and overfitting.

[Overfitting](https://www.ibm.com/topics/overfitting)

> When machine learning algorithms are constructed, they leverage a sample dataset to train the model. However, when the model trains for too long on sample data or when the model is too complex, it can start to learn the “noise,” or irrelevant information, within the dataset. When the model memorizes the noise and fits too closely to the training set, the model becomes “overfitted,” and it is unable to generalize well to new data. If a model cannot generalize well to new data, then it will not be able to perform the classification or prediction tasks that it was intended for.

[Underfitting](https://www.ibm.com/topics/underfitting)
> Underfitting is a scenario in data science where a data model is unable to capture the relationship between the input and output variables accurately, generating a high error rate on both the training set and unseen data.
> Underfitting occurs when a model is too simple, which can be a result of a model needing more training time, more input features, or less regularization.

I completed the following steps to test the accuracy of the model:

Import the [test_train_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) method from scikit-learn along with KNeighbors Classifier and the labeled data DataFrame to [model_accuracy_test.py](/model_accuracy_testing.py).

Split the labeled data into a training set (80% of data) and test set (20% of data). Create an array of possible integer k-neighbor values from 1 to 12. For each k value, set up a KNeighborClassifier and fit the model. Compute the accuracy of the model on both the training data and the test data.

Print the test and training accuracy for each k value of neighbors and plot them using [matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html).

## Preprocessing and Cross-validation
To improve the reliabilidy of this model, I implemented the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) method to normalize distrubtion of the individual features. I then used [Grid Search Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to determine the ideal k value to prevent overfitting and underfitting.

## Conclusion
Prior to scaling this data this model predicted apprentice withdrawals with a test accuracy of 83%.  Implementing a data pipeline with Preprocessing and Cross-validation improved the model's accuracy to 98.37%.

Applying the model with a k of 13 to the unlabeled data set produces a list of 36 (of 1000) apprentices who are at risk of withdrawal.

This model could be trained on an organic dataset to make ongoing automated predictions of apprentice withdrawal risk, allowing apprenticeship providers to make targeted intervention to improve their completion rates.