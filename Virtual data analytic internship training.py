# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" ANZ virtual data analysis program """
'Step 0 import necessary package'
import pandas as pd 
import numpy as np

'Step 1: import the dataset + initial exploration'
file = "C:/Users/gn-dy/ANZ synthesised transaction dataset.csv"
dataset = pd.read_csv(file)
print(dataset.head())
dataset.info()
dataset.describe(include = 'all')
dataset.shape #number of columns and number of rows#
print(dataset.columns)


for c in columns:
    print(c, dataset[c].head(10))


def Categorical_characteris(dataset):
    columns = dataset.columns
    for c in columns:
        try:
            if dataset[c].dtype == object:
               print(c, ":", "Unique value is/are:", np.unique(dataset[c].values), "and", "Total number is:", len(np.unique(dataset[c].values)))
               print('\n')
        except:
            print(c, ":", "This is exception error in this case")
            print(dataset[c].head(3))
            print('\n')
            
Categorical_characteris(dataset)

columns = dataset.columns
for c in columns:
    if dataset[c].dtype == float:
        print(c)
        
columns = dataset.columns
for c in columns:
    print(c, ":", dataset[c].value_counts(sort=False))
    print("\n")

'Step 2: check if there is any missing value?'

dataset["bpay_biller_code"].isnull()
dataset["bpay_biller_code"].fillna(1, inplace=True)# this can fill in missing values with values#
dataset["bpay_biller_code"].head()
dataset.isnull().sum()

dataset["amount"].describe()
dataset["balance"].describe()

'Step 3: visualise basic count distribution in histogram'

from matplotlib import pyplot as plt

for c in columns:
    try:
        x = dataset[c]
        plt.hist(x, ec="red")
        plt.xlabel(c)
        plt.ylabel("count")
        plt.title(c)
        plt.show()
    except:
        print(c, ":", "is error in this case")
        


'Step 4 Contional selections in EDA for subset of data for comparision'    
# a series of questions can be answered by doing EDA#  

'Question 1 I want to see the transaction detials for each customer over period of time'
'and try to visual your result on the graph'
   
EDA = dataset[["first_name", "balance", "date", "amount", "movement"]]
EDA['first_name']
customers = np.unique(EDA['first_name'].values)
print(customers)
            

Customers = EDA.groupby('first_name')
Customers['balance'].plot()

import matplotlib.pyplot as plt

# gca stands for 'get current axis'
ax = plt.gca()
EDA.plot(kind='line',x='date',y='balance', color='first_name')
plt.show()
plt.line()



'Step 5 change the format of the date in python'






























"""Task 2, to bulid regression model to predict the annual salary for customers"""




""" Step 1 re-construct a new dataset, select the attribute that you think is important """
import pandas as pd

file = "C:/Users/gn-dy/ANZ Virtual Task 2.csv"

data = pd.read_csv(file)

data.keys()

'merchant_suburb', 'merchant_state'

temp_dataset = data[['amount', 'txn_description', 'gender', 'age', 'customer_id']]

temp_dataset = temp_dataset.iloc[:,][temp_dataset.txn_description == 'PAY/SALARY']

temp_dataset = temp_dataset.groupby(['customer_id', 'age', 'gender']).sum()

len(temp_dataset)
temp_dataset.head(60)

temp_dataset.to_csv("C:/Users/gn-dy/desktop/ANZ Virtual Task 2.csv")

# I created a new dataset here and save to my desktop
# export this new dataset



""" Step 2 find out the correlation and visualise it with scatter plot"""
import pandas as pd
import numpy as np
import seaborn as sns
file = "C:/Users/gn-dy/desktop/ANZ Virtual Task 2.testing.csv"
data = pd.read_csv(file)




# visualise the correlation with seaborn
sns.lmplot('age', 'amount', data = data, hue = 'gender')
data.corr()
"""
             age    amount
age     1.000000 -0.036504
amount -0.036504  1.000000
"""
"""from the graph, it can demonstrate a weak negative correlation between age and amount
however, there is slightly more negative correlation between gender(female) and amount as they become older"""


""" Step 3 get X(dependent variable) and y(independent variable) from dataset""" 


X = data['age'].values
y = data['amount'].values

"""
X_mean = X.mean()
y_mean = y.mean()
"""

""" Step 4 write a function for sample weight and bias calculation, this is the blueprint for fit function later"""

def weight_bias(X, y):
    result_above = 0
    result_below = 0
    X_mean = X.mean()
    y_mean = y.mean()
    for i in range(len(X)):
        result_above += (X[i] - X_mean)*(y[i] - y_mean) 
        result_below += (X[i] - X_mean)**2
    weight = result_above / result_below
    bias = y[0] - weight*X[0]
    return weight, bias


"""the formula for linear regression is below 
weight = (x - X_mean)(y - y_mean)  /  (x - X_mean)**2
then use y = w*X + b to work out the bias b
"""

""" Step 5 write a class object for linear regression regressor, it should have function of fit and predict"""

class linear_regressor():
    def __init__(self):
        self.weight = None
        self.bias = None 
        
# the fit function is actually the machine learning internal machanism
    def fit(self, X, y):
        result_above = 0
        result_below = 0
        X_mean = X.mean()
        y_mean = y.mean()
        for i in range(len(X)):
            result_above += (X[i] - X_mean)*(y[i] - y_mean) 
            result_below += (X[i] - X_mean)**2
        weight = result_above / result_below
        bias = y_mean - weight*X_mean
        # be careful here, not to use random point, use average point
        self.weight = weight
        self.bias = bias
    
    def predict(self, X):
        if self.weight == None:
            return "need to fit first"
        elif self.bias == None:
            return 'need to fit first'
        else:
            for value in X:
                y = self.weight * value + self.bias
                print(y)

LR1 = linear_regressor()
LR1.fit(X, y)
LR1.weight
LR1.bias   
X_test = np.array([23, 45, 34, 48, 29, 30])
y_predicted1 = LR1.predict(X = X_test)
             
"""
LR = linear_regressor()   
LR.fit(X, y)
y_predicted = LR.predict(X)

fit(X, y)

try X = 45, y = 12786.648636668535"""

""" However, cost function is needed to optimise the accuracy"""

"""def cost_function(X_predicted, X):"""


"""Mean Square Error, also mean R2, is the cost function for linear regression """
X_predicted = np.array([45])

for i in range(len(X)):
    total_error = 0
    total_error += (X[i] - X_predicted[i])**2
MSE = total_error / len(X)
"""MSE is mean squared error"""


""" Step 6 use Gradient descent method to optimise the process"""
    
"""Using Gradient descent method to optimise the weight and bias 
take derivative of MSE, with respect to weight and bias respectively, 
this will get two separated function for weight and bias 

technically, a graph for weight (all possible weight) against sum of MSE, and 
a graph for bias (all possible bias) against sum of MSE are rereranced 

a curve should be viewed, and when sum of MSE is at the lowest, 
the gradient of the curve on the graph should be zero
, the same for bias. 

so here, learning rate is applied to take baby step approach the minimum MSE
and number of learning rate optimisation is also applied"""

"""this machanism is the core of linear regression machine learning algorithm"""

"""
f′(weight,bias) = ⎡  ⎣df/d(weight)    df/d(bias)⎤  ⎦ 

here mean the function of MSE with respect to weight and bias"""


""" is equal to = """

"""
df/d(weight)    =      1N∑(the mean: just divide by len())    sum of −2xi * (yi−(mxi+b))
 
 
df/d(bias)   =    1N∑(the mean: just divide by len())    sum of  −2 * (yi−(mxi+b))

"""

""" when weight_deri and bias_deri are obtained, it times a small step called learning rate
and then update that to existing weight and bias

and use the updated weight and bias to calculate the new MSE again"""

class linear_regressor_with_optimisation():
    def __init__(self):
        self.weight = None
        self.bias = None
        
# the fit function is actually the machine learning internal machanism
    def fit(self, X, y, learning_rate, number_of_iteration):
        result_above = 0
        result_below = 0
        X_mean = X.mean()
        y_mean = y.mean()
        for i in range(len(X)):
            result_above += (X[i] - X_mean)*(y[i] - y_mean) 
            result_below += (X[i] - X_mean)**2
        weight = result_above / result_below
        bias = y_mean - weight*X_mean
        # Gradient descent kicks in here 
        
        weight_deri = 0
        bias_deri = 0
        for i in range(len(X)):
            weight_deri += ((-2*X[i]) * (y[i] - weight*X[i] - bias))
            bias_deri += -2 * (y[i] - weight*X[i] - bias)
        weight_deri = weight_deri / len(X)
        bias_deri = bias_deri / len(X)
        
        for times in range(number_of_iteration):
            weight -= weight_deri * learning_rate
            bias -= weight_deri * learning_rate
            # be careful here, use original weight and bias for minusing!!!!!!
        self.weight = weight
        self.bias = bias

    def predict(self, X):
        if self.weight == None:
            return "need to fit first"
        elif self.bias == None:
            return 'need to fit first'
        else:
            y = []
            for value in X:
                y.append(self.weight * value + self.bias)
            return y   



LR2 = linear_regressor_with_optimisation() 
LR2.fit(X, y, learning_rate = 0.01, number_of_iteration = 1000)
""" testing now"""
X_test = np.array([23, 45, 34, 48, 29, 30])
y_predicted2 = LR2.predict(X = X_test)

LR2.weight
LR2.bias


""" Step 7 how to valiate this linear regression model """


""" think what are the standard to test linear regression???
MSE ????   RMSE ????    R2 score???? 
        Does sklearn module have success matrix for linear regression????
        
        if so?? how to import ????

"""






"""Step 8 Manually construct a validation test for the created model"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


LR3 = linear_regressor_with_optimisation()
LR3.fit(X_train, y_train, learning_rate = 0.001, number_of_iteration = 50)
predicted_result = np.array(LR3.predict(X = X_test))


def R2_valiation(predicted_result = None, y_test = None):
    total_error = 0
    for i in range(len(y_test)):
        total_error += (y_test[i] - y_test.mean())**2
    
    explained_error = 0
    for i in range(len(predicted_result)):
        explained_error += (predicted_result[i] - y_test.mean())**2
    
    R_2_score = 1 - (explained_error / total_error) 
    return R_2_score

R2_valiation(predicted_result, y_test)

""" R2 score is 0.9164652458386935, this R2 score is between 0 to 1
the larger means this model explains most of the observed data points in dataset
in other words, higher accuracy"""






""" Step 9 challenge! to construct a CART decision tree model for this dataset """

import pandas as pd
import numpy as np
import seaborn as sns
file = "C:/Users/gn-dy/desktop/ANZ Virtual Task 2.testing.csv"
data = pd.read_csv(file)

data = data.drop(['customer_id'], axis = 1)

# separate data and target
data
target = data['amount']
variables = data.drop(['amount'], axis = 1)
A_person = data[0:1]


for value in data['gender'].unique():
    if value == 'F':
        index_f = data['gender'] == value
    else:
        index_m = data['gender'] == value        
data[index_f]    
data[index_m]  


class Simple_CART_decision_tree():
    def __init__(self):
        self.regression = None
        self.children = {}
        self.weight_F = None
        self.bias_F = None
        self.weight_M = None
        self.weight_M = None
        

    def fit(self, X, y):
        for value in X['gender'].unique():
            if value == 'F':
                index = X['gender'] == value
                self.children[value] = X[index]
                LR_F = linear_regressor_with_optimisation()
                age = np.array(self.children[value]['age'])
                amount = np.array(self.children[value]['amount'])
                LR_F.fit(X = age, y = amount, learning_rate = 0.001, number_of_iteration=50)
                self.weight_F = LR_F.weight
                self.bias_F = LR_F.bias
            
            else:
                index = X['gender'] == value
                self.children[value] = X[index]
                LR_M = linear_regressor_with_optimisation()
                age = np.array(self.children[value]['age'])
                amount = np.array(self.children[value]['amount'])
                LR_M.fit(X = age, y = amount, learning_rate = 0.001, number_of_iteration=50)
                self.weight_M = LR_M.weight
                self.bias_M = LR_M.bias
     
    def predict(self, X):
          if self.weight_F is None:
             return "need to fit first"
          elif self.bias_F is None:
             return 'need to fit first'
          elif self.weight_M is None:
             return "need to fit first"
          elif self.bias_M is None:
             return 'need to fit first'
         
          for value in X['gender']:
              if value == 'F':
                  index = X['gender'] == value
                  age_F = X[index]['age'].values
              else:
                  index = X['gender'] == value
                  age_M = X[index]['age'].values
          y_M = []
          for value in age_M:
              y_M.append(self.weight_M * value + self.bias_M)
          y_F = []
          for value in age_F:
              y_F.append(self.weight_F * value + self.bias_F)
          return y_F, y_M


dt = Simple_CART_decision_tree()
dt.fit(variables, target, data)
dt.children
dt.weight_F
dt.bias_F
dt.weight_M
dt.bias_M

data
train = data
test = data.drop(['amount'], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.3, random_state = 42)

dt2 = Simple_CART_decision_tree()
dt2.fit(X_train, y_train)
result = dt2.predict(X_test)

predicted_result_Female = result[0]
predicted_result_Male = result[1]

print(predicted_result_Female)
print(predicted_result_Male)

# be careful we use the amount column in X_test here as for testing data 
"""
for_female_test = X_test[:13]['amount'].values
for_male_test = X_test[-17:]['amount'].values
"""
# note here, need to know how to get the first XX values and the last XX values from the data

"""# for fairness, randomly select 13 and randomly select 17 are better""" 
import random 
for_female_test = np.array(random.choices(X_test['amount'].values, k = 13))
for_male_test = np.array(random.choices(X_test['amount'].values, k = 17))


R2_valiation(predicted_result = predicted_result_Female, y_test = for_female_test)
""" result = 0.9765017635237022 """
R2_valiation(predicted_result = predicted_result_Male, y_test = for_male_test)
""" result = 0.969825795385937 """

""" compare to previous R2 score of 0.91.... this model has higher score"""