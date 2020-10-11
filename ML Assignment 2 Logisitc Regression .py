import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.05, n_iterations=250):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent
        for _ in range(self.n_iterations):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self.sigmoid(linear_model)
            # compute gradients
            derivative_weights = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            derivative_bias = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights = self.weights - self.learning_rate * derivative_weights
            self.bias = self.bias - self.learning_rate * derivative_bias

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
                    
                    
                    
                    
"""Model Evaluation testing scheme"""

from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

"""Using wine dataset to evaluate accuracy"""
dataset1 = datasets.load_wine()
X, y = dataset1.data, dataset1.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)

regressor = LogisticRegression(learning_rate=0.05, n_iterations=250)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("Logistic regression accuracy rate is:", accuracy(y_test, predictions))
"""End of test one"""

"""Using breat_cancer dataset to evaluate accuracy"""
dataset2 = datasets.load_breast_cancer()
X, y = dataset2.data, dataset2.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)

regressor = LogisticRegression(learning_rate=0.05, n_iterations=250)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("Logistic regression accuracy rate is:", accuracy(y_test, predictions))
"""End of test two"""



"""Using wine dataset to evaluate accuracy with 0.0001 learning rate"""
dataset1 = datasets.load_wine()
X, y = dataset1.data, dataset1.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)

regressor = LogisticRegression(learning_rate=0.0001, n_iterations=250)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("Logistic regression accuracy rate is:", accuracy(y_test, predictions))
"""End of test three"""

"""Using breat_cancer dataset to evaluate accuracy with 0.0001 learning rate"""
dataset2 = datasets.load_breast_cancer()
X, y = dataset2.data, dataset2.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)

regressor = LogisticRegression(learning_rate=0.0001, n_iterations=250)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("Logistic regression accuracy rate is:", accuracy(y_test, predictions))
"""End of test four"""