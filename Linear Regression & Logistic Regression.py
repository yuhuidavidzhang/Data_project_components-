import numpy as np
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import math 


from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

# Sometime, we may not want to see abbrivated data, we can use set_option, to expand the columns.
pd.set_option('display.max_columns', None)


# May use info() function to initially explore the data 
#train_set.info()
#test_set.info() 

# For some descriptive statistics, we can use info(), corr(), describe() or summary()

# look at numeric data and categorical data separately 

"""Step 1 load the dataset"""

path = "C:/Users/gn-dy/birthwt2020.csv"
data_set = pd.read_csv(path)
print(data_set.head())

"""Step 2 select the column i want for linear regression"""
data_set = df[['lwt', 'bwt']]
print(data_set.head())

"""Step 3 Data Cleaning process"""
data_set.dtypes  """what data type is?"""
data_set.isna().any()   """check if there is any missisng value at all?"""

"""if Ture for missing value, use dropna() to drop"""

"""Step 4 Using basic visualisation tool to do initial visual data analysis""" 
x = data_set['lwt']
y = data_set['bwt']

plt.plot(x, y, color = 'blue')

plt.title('Relationship between birth weight and mothers weight')
plt.xlabel('mothers weight in pounds')
plt.ylabel('brith weight in grams')
plt.legend()
plt.show()


"""Step 5 Check the correlation between these two variables"""
data_set.corr()

"""Step 6 Take a look statistic summary"""
data_set.describe()


"""Step 7 Is there any outliers or hava a look for skewness"""
data_set.hist(grid = False, color = 'orange')   """Create the histogram"""

"""Step 8 Check for data dimension shape or change the shape of data for model fitting"""
x.ndim
y.ndim
x.shape
y.shape

x = np.array(x)
print(x)
y = np.array(y)
print(y)


x = x.reshape(-1,1)
x.ndim
x.shape

y = y.reshape(-1,1)
y.ndim
y.shape

"""check the dimension and the shape and convert 1D array to 2D array"""


""" Step 9 Build a model"""

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=1)


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

plt.scatter(x, y, color = 'red')
plt.plot(x, regression_model.predict(x), color = 'blue')
plt.title('The Liner regression result Demo')
plt.xlabel('mothers weight in pounds')
plt.ylabel('birth weight in gram')
plt.show()


intercept = regression_model.intercept_[0]
print(intercept)

coefficient = regression_model.coef_[0]
print(coefficient)


"""Step 10 Make some prediction using the trained model"""
predict_values = regression_model.predict(x)
print(predict_values)



"""Step 11 Validate the mdoel by assessing the model fitness"""
"""Need to calculate 
Mean Absolute Error: 
Mean Squared Error: MSE punishes large errors
Root mean squared error: """

R-squared

regression_model_R2 = r2_score(y, predict_values)
print(regression_model_R2)










