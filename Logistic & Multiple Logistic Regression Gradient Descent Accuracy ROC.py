# -*- coding: utf-8 -*-

from sklearn import datasets
data = datasets.load_iris()

import pandas as pd
import numpy as np

""" do a manual minmax scalar 

x - x.min     /     x.max - x.min
"""


index = data['target'] == 0
temp_data = data['data']



X = (temp_data - temp_data.min()) / (temp_data.max() - temp_data.min())


X = pd.DataFrame(data['data'])
y = data['target']



# might be good idea to normalise the data before fit
# normalise features might be also applied to linear regression




"""
temp_dataframe = pd.DataFrame(data['data'], columns = ['F1', 'F2', 'F3', 'F4'])
temp_dataframe.corr()
import matplotlib.pyplot as plt
plt.matshow(temp_dataframe.corr())      
plt.show()


the above is to visualise correlation


"""




class Logistic_regression_model():
    def __init__(self):
        self.weights = None
        self.bias = None
        self.classifiers_thetas = None
        self.classifiers_bias = None
        
        
    def binary_fit(self, X, y, learning_rate = None, number_of_steps = None):  
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(number_of_steps):
            lm = np.dot(X, self.weights) + self.bias # be careful, the order here does matter, think of 22 and 24
            deri_w = (1 / n_samples) * np.dot(X.T, (y - self.sigmoid_function(lm)))
            deri_b = (1 / n_samples) * np.sum(y - self.sigmoid_function(lm))
            self.weights -= deri_w * learning_rate
            self.bias -= deri_b * learning_rate
        return np.array(self.weights), np.array(self.bias)
        

    def multiclass_fit(self, X, y, learning_rate = None, number_of_steps = None):     
        for i in np.unique(y):
            index = data['target'] == i
            temp_y = np.where(index == True, 1, index)
            multiclass_thetas_bias = self.binary_fit(X, temp_y, learning_rate, number_of_steps)
            self.classifiers_thetas = np.append(self.classifiers_thetas, multiclass_thetas_bias[0])
            self.classifiers_bias = np.append(self.classifiers_bias, multiclass_thetas_bias[1])
        self.classifiers_thetas = np.delete(self.classifiers_thetas, 0, axis = 0)
        self.classifiers_bias = np.delete(self.classifiers_bias, 0, axis = 0)
        self.classifiers_thetas = self.classifiers_thetas.reshape(len(np.unique(y)), len(X.keys()))
        self.classifiers_bias = self.classifiers_bias.reshape(-1, 1)  
        
    def cost_function():
        pass
        
    def loss_function():
        # Linear regression uses least square error to measure its loss value
        pass
        
    def sigmoid_function(self, linear_values):
        logistic_values = 1 / (1 + np.exp(-linear_values))
        return logistic_values
        
    def binary_predict(self, X):
        if self.weights is None:
            print("Warning, you need to train the model first!!!!")
        else:
            y_predicted = np.dot(X, self.weights) + self.bias
            logistic_values = 1 / (1 + np.exp(-y_predicted))
            #logistic_Value = [1 if i >= 0.5 else 0 for i in logistic_Value] 
            # be careful with the cutting point for classification (the threshold value)
            return logistic_values
        
    def multiclass_predict(self, X):
        if self.classifiers_thetas is None:
            print("Warning, you need to train the model first!!!!")
        else:
            for i in range(len(X)):
                y_predicted = np.dot(X.values[i].reshape(-1, 1), self.classifiers_thetas) + self.classifiers_bias
                #logistic_values = 1 / (1 + np.exp(-y_predicted))
                #record_list.append(logistic_values.max())
                print(y_predicted)


LogR = Logistic_regression_model()
LogR.binary_fit(X, y, learning_rate = 0.0001, number_of_steps = 250)
LogR.multiclass_fit(X, y, learning_rate = 0.0001, number_of_steps = 250)
LogR.weights
LogR.bias
LogR.classifiers_thetas
LogR.classifiers_bias

LogR.binary_predict(X)
LogR.multiclass_predict(X)

i = 130
np.dot(X.values[i], LogR.classifiers_thetas[0]) + LogR.classifiers_bias[0]
np.dot(X.values[i], LogR.classifiers_thetas[1]) + LogR.classifiers_bias[1]
np.dot(X.values[i], LogR.classifiers_thetas[2]) + LogR.classifiers_bias[2]






""" might need copy and paste the context here """

class Multiclass_Logistic_regression():
    def __init__(self):
        self.weights = None
        self.bias = None
        self.number_of_class = None
        self.learning_rate = 0.05
        self.number_of_steps = 500
        
    def fit(self, X, y):  
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.number_of_class = len(np.unique(y))
        self.y = y


        
        for i in range(self.number_of_steps):
            lm = np.dot(X, self.weights) + self.bias # be careful, the order here does matter, think of 22 and 24
            # Sigmoid function
            logistic_Value = 1 / (1 + np.exp(-lm))
            deri_w = (1 / n_samples) * np.dot(X.T, (y - logistic_Value))
            deri_b = (1 / n_samples) * np.sum((y - logistic_Value))
            self.weights -= deri_w * self.learning_rate
            self.bias -= deri_b * self.learning_rate
        
    def predict(self, X):
        if self.weights is None:
            print("Warning!" "\n" "needs to fit first")
        else:
            y_predicted = {}
            for value in np.unique(y):
                temp_array = np.where(self.y != value, -1, self.y)
                self.fit(X, temp_array)
                y_predicted = np.dot(X, self.weights) + self.bias
                logistic_Value = 1 / (1 + np.exp(-y_predicted))
                y_predicted[value] = logistic_Value.max()
            return y_predicted
                
            
        else:
            y_predicted = np.dot(X, self.weights) + self.bias
            logistic_Value = 1 / (1 + np.exp(-y_predicted))
            #logistic_Value = [1 if i >= 0.499845 else 0 for i in logistic_Value] # be careful with the cut point for classification
            return logistic_Value
        
L = Multiclass_Logistic_regression()
L.fit(X, y)
L.predict(X)
for value in np.unique(y):
    temp_array = np.where(y != value, -1, y)
    clrs = L.fit(X, temp_array, learning_rate=0.05, number_of_steps=1000)
    

















""
Graph visualisation 


import matplotlib.pyplot as plt

plt.plot(LogR.binary_predict(X))
plt.show()

plt.hist(LogR.predict(X))
plt.show()






""" Logistic regression model evaluation method 


from sklearn import metrics


metrics.roc_auc_score(args, y_true, y_score)
metrics.accuracy_score(args, y_true, y_pred) 
metrics.confusion_matrix(y_true, y_pred)


"""

""" the below code is my assignment 2 for machine learning UTS """

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
        #y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
temp_X = X.iloc[100: -30]       
L = LogisticRegression()
L.fit(X, temp_y)
L.predict(temp_X)