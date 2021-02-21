# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 01:00:31 2021

@author: gn-dy
"""



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







""" Multiple Linear Regression"""

""" Step 1 get the dataset"""

import pandas as pd
import numpy as np
import seaborn as sns
file = "C:/Users/gn-dy/ANZ Virtual Task 2.testing.csv"
data = pd.read_csv(file)

"""
       customer_id  age gender    amount
0   CUS-1005756958   53      F  12616.11
1   CUS-1117979751   21      M  25050.55
2   CUS-1140341822   28      M  11499.06
3   CUS-1147642491   34      F  22248.07
4   CUS-1196156254   34      F  27326.11
"""


""" Step 2 Separate gender as dummies variable"""

data = pd.get_dummies(data, columns = ['gender'])
"""
       customer_id  age    amount  gender_F  gender_M
0   CUS-1005756958   53  12616.11         1         0
1   CUS-1117979751   21  25050.55         0         1
2   CUS-1140341822   28  11499.06         0         1
3   CUS-1147642491   34  22248.07         1         0
4   CUS-1196156254   34  27326.11         1         0
"""

""" Step 3 Split training set and target set, check the shape
the shape is very important as matrix operation will be involved later """

data = data.drop(['customer_id'], axis = 1)


data_independent = data.drop(['amount', 'gender_M'], axis = 1) # convert to np.array

# be very careful about the dummy variable trap !!!!
# if two dummy variables are related, and one predict to another for example gender
# please remove that variable columns 

data_dependent = data['amount'] # convert to np.array

data_independent.shape #[100, 3] can be seen as 100X3 matrix, three features, three thetas values
data_dependent.shape #[100,] but can be seen as 100X1 matrix 


# y = theta * X + error
# but in multiple linear regression, this means matrix calculations
"""
[y,         [x1.....x....             [theta1,                [error]
 y1   *      x......x.....       *     theta2,       + 
 y2          x......x.....             theta3,
 y3]         x......x.....]            theta4]



theta = (y - error) * 1/X 

theta = (X.transpose * X)**(-1) * X.transpose*y

"""
class Multiple_linear_regression():
    def __init__(self):
        self.pre_weights = None
        self.bias = None
        self.pre_thetas_details = None
        self.post_weights = None
        self.thetas = None
    
    def Gradient_descent_optimisation(self, X, y, learning_rate = None, number_of_steps = None):
        if self.pre_weights is None:
            print("Needs to fit the model first")
        else:
            array = np.full((len(X), 1), 1)
            temp_X = np.append(X, array, axis = 1)
            y_predicted_pre = temp_X.dot(self.pre_weights).reshape(-1, 1)
            gap = y.values.reshape(-1, 1) - y_predicted_pre
            v = len(X.keys())
            n = v
            gap = y.values.reshape(-1, 1) - y_predicted_pre
            deri_X = []
            for i in range(len(X.keys())):
                deri_X.append(gap * (-(X.iloc[:, (v - n)].values.reshape(-1, 1))))
                n = n - 1
                if n == 0:
                    break
            # Gradient Descent for MLR comes in here
            k = 0
            while k < number_of_steps: # interesting while loop, and for loop inside this while loop
                for i in range(len(X.keys())):
                    self.thetas[i] -= deri_X[i].mean() * self.thetas[i] * learning_rate
                k = k + 1
            self.post_weights = self.thetas

        
    def fit(self, X, y):
        features_list = []
        for i in X.keys():
            features_list.append(i)
        features_list.append('bias')        
        array = np.full((len(X), 1), 1)
        X = np.append(X, array, axis = 1)       
        n_sample, n_features = X.shape
        n_forloops = n_features
        y = y.values
        XT_X = (X).T.dot(X) # X times X.transpose = X.transpose * X
        XT_X_inverse = np.linalg.inv(XT_X)     # inverse of (X.transpose * X)
        XT_y = (X).T.dot(y)   # X.transpose times y
        thetas = XT_X_inverse.dot(XT_y)   # times together for both
        self.thetas = thetas
        self.pre_weights = thetas
        self.bias = thetas[-1]
        self.pre_thetas_details = {}
        for i in np.array(features_list):
            self.pre_thetas_details[i] = thetas[n_features - n_forloops]
            n_forloops = n_forloops - 1  # here create a loop to generate thetas details in dictionary
            if n_forloops == 0:
                break
              
    def predict(self, X):
        if self.post_weights is None:           
            array = np.full((len(X), 1), 1)
            temp_X = np.append(X, array, axis = 1)
            y_predicted_pre = temp_X.dot(self.pre_weights).reshape(-1, 1)
            return y_predicted_pre
        else:
            array = np.full((len(X), 1), 1)
            temp_X = np.append(X, array, axis = 1)
            y_predicted_post = temp_X.dot(self.post_weights).reshape(-1, 1)
            return y_predicted_post
            


"""note that matrix multiplication is achieved by dot()"""

X = data_independent
y = data_dependent


MLR = Multiple_linear_regression()
MLR.fit(X, y)
MLR.pre_thetas_details
MLR.pre_weights
MLR.predict(X)
MLR.bias
MLR.Gradient_descent_optimisation(X, y, learning_rate = 0.01, number_of_steps = 10000)
MLR.post_weights




"""Multiple/ Linear regression using purely Gradient Descent approach"""

class LinearRegression_GD:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.lm = None
        
    def fit(self, X, y, learning_rate = None, number_of_iteration = None):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(number_of_iteration):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y))  # why there is no 2 in front of n_sample
            db = (1 / n_sample) * np.sum(y_predicted - y)      # why there is no 2 in front of n_sample 
            self.weights -= dw * learning_rate
            self.bias -= db * learning_rate
     
    def predict(self, X):
        if self.weight is None:
            return 0
        else:
            y_approximated = np.dot(X, self.weights) + self.bias
        return 
        
# purely use gradient descent approach, the outcome are very sensitive to learning rate and numner of iterations
                
LR = LinearRegression_GD()
LR.fit(X, y, learning_rate = 0.001, number_of_iteration = 1000)
LR.weights
LR.bias

"""


total_error = 0
for i in range(len(y_predicted)):
    total_error += (data_test[i] - data_test.mean())**2

explained_error = 0
for i in range(len(y_predicted)):
    explained_error += (y_predicted[i] - data_test.mean())**2

R2 = 1 - (explained_error / total_error)
print("R2: ", R2)



y_pred = (X.values).dot(self.pre_weights).reshape(-1, 1)
        total_error = 0
        for values in range(len(y)):
            total_error += (y[values] - y_pred[values])**2
        self.pre_bias = total_error / (2*len(y))
        
"""