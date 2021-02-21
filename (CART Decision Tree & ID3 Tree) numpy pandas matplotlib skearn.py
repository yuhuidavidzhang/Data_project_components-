# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 21:29:29 2020

@author: gn-dy
"""
"""Step 1 import the dataset"""
from sklearn import datasets
data = datasets.load_iris()
print(data)

"Step 2 examine the dataset"
print(type(data['data']))
print(type(data['target']))

data['data'].shape
data['target'].shape

# seperate dataset into data set and target set 
data['data']
data['target'].shape

"Step 3 re-structure and re-name columns for the dataset"
import pandas as pd 

data_set_features = pd.DataFrame(data['data'], columns=('F1', 'F2', 'F3', 'F4'))
print(data_set_features)
data_set_features.shape

data_set_target = pd.DataFrame(data['target'])
data_set_target.columns = ['target']
print(data_set_target)
data_set_target.shape

data = pd.concat([data_set_features, data_set_target], axis = 1)
print(data)
data.shape


#from here still use data as reference!!!!!!!

# think use isin()
        

"Step 4 categories numerial data into bins, and write the key function for CART Tree"

for column in data.drop(['target'], axis = 1).columns:
    data[column] = pd.cut(data[column], 3, labels = ("low", "mid", "high" ))


# seperate into train set and target set 
X = data.drop(['target'], axis = 1)
y = data['target']
    


"""for column in X_train.columns:
        X_train[column] = pd.cut(X_train[column], 3, labels = ("low", "mid", "high" ))
"""
""" you might un-label first to see the range for each bin
for column in X.columns:
    print(X[column].unique())
    
trial_data = X.sample(5)
print(trial_data)
"""
"""
        feature_1     feature_2       feature_3      feature_4  target
55     (5.5, 6.7]  (1.998, 2.8]  (2.967, 4.933]     (0.9, 1.7]       1
145    (5.5, 6.7]    (2.8, 3.6]    (4.933, 6.9]     (1.7, 2.5]       2
26   (4.296, 5.5]    (2.8, 3.6]  (0.994, 2.967]  (0.0976, 0.9]       0
91     (5.5, 6.7]    (2.8, 3.6]  (2.967, 4.933]     (0.9, 1.7]       1
33   (4.296, 5.5]    (3.6, 4.4]  (0.994, 2.967]  (0.0976, 0.9]       0
""" 

"""
"feature_1 : 4.296 - 5.5 is low/ 5.5 - 6.7 is mid/ 6.7 - 7.9 is high"
"feature_2 : 1.998 - 2.8 is low/ 2.8 - 3.6 is mid/ 3.6 - 4.4 is high"
"feature_3 : 0.994 - 2.967 is low/ 2.967 - 4.933 is mid/ 4.933 - 6.9 is high"
"feature_4 : 0.00976 - 0.9 is low/ 0.9 - 1.7 is mid/ 1.7 - 2.5 is high"

"""
print(X) # as data input (for training set)
print(y) # as data output (for target set)

""" Step 5 write a loss function to represent its statisical learning """


def gini_index(column, X, y):
    for v in X[column].unique():
        if v == 'low':
            index_low = X[column] == v 
        elif v == 'mid':
            index_mid = X[column] == v
        elif v == 'high':
            index_high = X[column] == v 
# First to fliter out the corresponding target value to each category in this column
    result_low = 0
    for value in y[index_low].value_counts():
        result_low += (value/ len(y[index_low]))**2
        result_low = 1 - result_low               
    result_mid = 0
    for value in y[index_mid].value_counts():
        result_mid += (value/ len(y[index_mid]))**2
        result_mid = 1 - result_mid                
    result_high = 0
    for value in y[index_high].value_counts():
        result_high += (value/ len(y[index_high]))**2
        result_high = 1 - result_high               
# Second to calculate gini value of each category  
    for v, fr in X[column].value_counts(sort = False).iteritems():
        if v == 'low':
            result_low = result_low * (fr/len(X[column]))
        elif v == 'mid':
            result_mid = result_mid * (fr/len(X[column]))
        elif v == 'high':
            result_high = result_high * (fr/len(X[column]))    
    gini_index = result_low + result_mid + result_high
    return gini_index
# Third to square and sum them to work out the gini index for this column












" Step 6 Create a class object for decision tree that has function of fit and predict, maybe validate as well"

# this function takes training array and target array into it

class TreeNode():
    def __init__(self):
        self.spliting_feature = None
        self.children = {}
        self.decision = None
      

    def fit(self, X, y):
        if X == 0:
            self.decision = 'should be ok now?!?!'
            return
        else:
            unique_value = y.unique()
            len(unique_value) == 1
            self.decision = unique_value[0]
            
        spliting_index = 1
        for v in X.keys():
            if spliting_index > gini_index(v, X, y):
                spliting_index = gini_index(v, X, y)
                spliting_feature = v 
        for a in X[spliting_feature].unique():
            index = X[spliting_feature] == a
            self.children[a] = TreeNode()
            self.children[a].fit(X[index], y[index])        
         


dt = TreeNode()
dt.fit(X,y)






def feature_noise(column, X, y):
    column_ratio = []
    for i in X[column].value_counts():
        column_ratio.append(i/ len(y))    
    column_gini_low = []
    column_gini_mid = []
    column_gini_high = []
    column_gini = []
    for target in y.unique():
        column_gini_low.append((len(X.loc[(X[column] == 'low') & (y == target)])/ len(y))**2) 
        column_gini_mid.append((len(X.loc[(X[column] == 'mid') & (y == target)])/ len(y))**2)                   
        column_gini_high.append((len(X.loc[(X[column] == 'high') & (y == target)])/ len(y))**2)                   
    gini_low = 1 - sum(column_gini_low)
    gini_mid = 1 - sum(column_gini_mid)
    gini_high = 1 - sum(column_gini_high)
    column_gini = [gini_low, gini_mid, gini_high]
    gini_noise = [a * b for a, b in zip(column_gini, column_ratio)]
    return sum(gini_noise)

spliting_feature = None
spliting_index = 1
decision = None  
for v in X.keys(): 
    if spliting_index > feature_noise(v, X, y):
        spliting_index = feature_noise(v, X, y)
        spliting_feature = v 





""' this is optional for y'
y = y.replace({2: 1})
y.unique()
y.value_counts()
 

""" Now let try the ID3 algorithm"""

import numpy as np 

def entropy_of_target(y):
    values = np.array(y.value_counts(normalize = True))
    return - (values * np.log2(values)).sum()

def entropy_for_column(column, X, y):
    for value in X[column].value_counts(sort = False):
        return ((value/ len(y)) * entropy_of_target(y)).sum()

class TreeNode():
    def __init__(self):
        self.spliting_feature = None
        self.children = {}
        self.decision = None
      
    def fit(self, X, y):
        if len(X) < 20:
            self.decision = 'should be ok now?!?!'
            return
        else:
            unique_value = y.unique()
            if len(unique_value) == 1:
                self.decision = unique_value[0]
            else:
                info_gain = 0
                for column in X.keys():
                    if info_gain < (entropy_of_target(y) - entropy_for_column(column, X, y)):
                        info_gain = entropy_of_target(y)- entropy_for_column(column, X, y)
                        spliting_feature = column
                for a in X[spliting_feature].unique():
                    index = X[spliting_feature] == a
                    self.children[a] = TreeNode()
                    self.children[a].fit(X[index], y[index])        
         
dt = TreeNode()
dt.fit(X, y)

_raw_data = """Outlook,Temperature,Humidity,Wind,Play
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Overcast,Cool,Normal,Strong,Yes
Sunny,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Rain,Cool,Normal,Weak,Yes
Sunny,Mild,Normal,Weak,Yes
Overcast,Mild,Normal,Strong,Yes
Overcast,Hot,Normal,Strong,Yes
"""
with open("sport_data.csv", "w") as f:
    f.write(_raw_data)
df = pd.read_csv("sport_data.csv")
X = df.drop(['Play'], axis = 1)
y = df['Play']

