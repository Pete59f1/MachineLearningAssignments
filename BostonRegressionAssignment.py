import pandas as pd
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Looking at the current dataset
# Looking at the dataset, we can see that we have three arrays: data, target and feature_names
bostonData = load_boston()
# print(bostonData)

# Getting our feature data from the array called data using the feature_names array to get the right columns
X = pd.DataFrame(bostonData.data, columns=bostonData.feature_names)
# Getting out label data from the array called target
y = pd.DataFrame(bostonData.target)

# print(X.describe())
# print(y.describe())

# Splitting our features and labels into training and testing data
# Setting our test size to 25% of the data
# Setting our random state to 5, so training and testing data is always the same,
# No matter how many times we execute the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Creating and training model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Printing how well our model did
print(model.score(X_test, y_test))
