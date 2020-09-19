import numpy as np
import pandas as pd
import seaborn as sb 
import matplotlib as mp 

# Sometime, we may not want to see abbrivated data, we can use set_option, to expand the columns.
pd.set_option('display.max_columns', None)

train_set = pd.read_csv("C:/Users/gn-dy/train.csv")
test_set = pd.read_csv("C:/Users/gn-dy/test.csv")

# May use info() function to initially explore the data 
#train_set.info()
#test_set.info() 

# For some descriptive statistics, we can use info(), corr(), describe() or summary()

# look at numeric data and categorical data separately 

train_set.info()