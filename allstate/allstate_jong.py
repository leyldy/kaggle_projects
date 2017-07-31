
# coding: utf-8

# # ALLSTATE CLAIMS SEVERITY

# In[33]:

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#import seaborn as sns
import math
from sklearn import neighbors, datasets, ensemble, cross_validation, utils
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.cross_validation import cross_val_score


# ## EXPLORATORY DATA ANALYSIS

# First, we will find how many unique values there are in each column containing "cat#", and list out in an array what those unique values are. Hopefully by these counts we are able to find some type of pattern!

# In[49]:

#Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:

#Find unique values
cat_names = list(train)
for name in cat_names :
    if name.find("cat") != -1 :
        print(name, ":", train[name].unique(), "Total unique values: ", len(train[name].unique()))


# From this analysis we can speculate:
# 
# * columns "cat1" through "cat72" might be True/False or Yes/No
# 
# * columns "cat73" through "cat76" might be Low/Medium/High values, such as income class or the like with 3 categories
# 
# * columns "cat77" through "cat88" also seem to be category-based, with 4 categories each
# 
# * specifically for column 112, it may represent states as there is 51 unique values.

# Now, let's also check some statistics about quantitative variables.

# In[4]:

#Basic Statistics
train.describe()


# In[5]:

#Skewness of Continuous Variables
train.skew()


# We can clearly see there is high skew factor in loss - let's check what the histogram of it looks like.

# In[6]:

# histogram of loss
train['loss'].hist(bins=1000)


# In[7]:

# looks not very normal. let's log(1+x) it
np.log1p(train['loss']).hist(bins=1000)


# In[28]:

#How about cont9, the highest-skewed feature?
train['cont9'].hist(bins=100)


# In[29]:

#Try long(1+x) to see if there is major improvements?
np.log1p(train['cont9']).hist(bins=100)


# Skew of the loss function was much improved through log(1+x)'ing it, but not cont9, the highest-skewed feature. So for now let's keep in mind to log(1+x) the loss variable (our response variable).

# ## FEATURE ENGINEERING

# First, let's do the simple things: Unskew loss:

# In[52]:

train['loss'] = np.log1p(train['loss'])


# Now, let us fill in the missing variables in continuous variables, with their means.

# In[5]:

#Just checking: Are any values missing?
train.isnull().values.any()


# Since no values are missing, no need to fill in NAN's with means.

# In[15]:

#No values missing, no need to do this:
#train = train.fillna(train.mean())


# Let's see which variables are correlated/unnecessary variables and drop them:

# In[44]:

#First only getting continuous variable columns
cont_columns = []

for i in train.columns:
    if train[i].dtype == 'float':
        cont_columns.append(i)
cont_columns


# In[11]:

train.loc[:, cont_columns].corr()


# Correlated Variables:
# * cont1/cont9
# * cont1/cont10
# * cont6 with a lot of variables, just remove
# * cont7 with cont11/12
# * cont9 with cont10
# * cont10 with a lotta variables
# * cont 11 with cont12
# * cont6 with cont13

# In[6]:

#This code was taken and modified from Kaggle user "denoiser"'s kernel "Simple EDA - feature transformations"

# Compute the correlation matrix
corr = train[cont_columns].corr()

sns.set(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# By the looks of the previous two analyses, the most uncorrelated 'cont' variables that would serve as better predictors are: 
# 
# * cont2, cont3, cont4, cont5, and cont14 

# In[53]:

#Knock out the overly correlated columns
unwanted_list = ['id', 'cont1','cont6','cont7','cont8','cont9','cont10', 'cont11','cont12','cont13']
train.drop(unwanted_list, axis = 1, inplace = True)


# In[54]:

#For some reason the above code produces an error, but the action is done here anyways, 
#as verified by taking a peek at the new training dataset
train.shape


# We took a look at the recommended transformations on categorical variables given in the kernel "Simple EDA - feature transformations"by user 'denoiser', and applied these transformations on the training set in the cont categories that we kept.

# In[55]:

#Apply appropriate transformations on the columns we want -- gosh, I sound like a commercial lol
from scipy import stats

train['cont2'] = np.tan(train['cont2'])
train['cont4'] = stats.boxcox(train['cont4'])[0]
train['cont5'] = stats.boxcox(train['cont5'])[0]
train.head(3)


# Next, let's create one-hot encodings for categorical variables.

# In[56]:

#drop_first = True to remove perfect multicollinearity
train = pd.get_dummies(train, drop_first=True)
train.shape


# ## PREPARING DATA

# Preparing data as necessary through cross-validation split.

# In[57]:

X = train.drop('loss', axis = 1)
Y = train['loss']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=123)


# ## LINEAR REGRESSION

# In[15]:

from sklearn import linear_model


# In[62]:

#Just checking the shape of train and test shape for validation
print("Local train shape: ", X_train.shape)
print("Local test shape: ", X_val.shape)


# Now let us run the model!

# In[17]:

clf = linear_model.LinearRegression()
clf.fit(X_train, Y_train)


# Let's cross validate the model and see if it's robust.

# In[41]:

#Function for cross-validation scores
def cv_stats(cv_score):
    """ Returns the mean and standard deviation in a readable format"""
    mean = np.mean(cv_score)
    std = np.std(cv_score)
    return mean, std


# In[45]:

clf_score = cross_val_score(clf, X_train, Y_train, cv = 5)
cv_stats(clf_score)


# Let's try running this model for local_test or local validation.

# In[18]:

local_y_pred = clf.predict(X_val)


# Converting log(1+x)'ed scores back into actual loss response variables to check for MAE.

# In[22]:

local_lin_mae = mean_absolute_error(np.expm1(Y_val), np.expm1(local_y_pred))
local_lin_mae


# ## RANDOM FOREST

# Linear Regression ain't good enough, so Random Forest it is: recall that from cat112, we believe that the data can be separated by state (51, including DC). So let's set the n_estimators to 51.

# In[41]:

from sklearn.ensemble import RandomForestRegressor


# In[ ]:

rf = RandomForestRegressor(n_estimators=500, random_state = 123)
rf.fit(X_train,Y_train)
preds = rf.predict(X_val)
preds[0:5]


# In[16]:

local_rf_mae = mean_absolute_error(np.expm1(Y_val), np.expm1(preds))
local_rf_mae


# ## MAKING A SUBMISSION

# Feature Engineering Test Data:

# In[69]:

test = pd.read_csv('test.csv')
ID = test['id']
unwanted_list = ['id','cont1','cont6','cont7','cont8','cont9','cont10', 'cont11','cont12','cont13']
test.drop(unwanted_list, axis = 1, inplace = True)
test['cont2'] = np.tan(test['cont2'])
test['cont4'] = stats.boxcox(test['cont4'])[0]
test['cont5'] = stats.boxcox(test['cont5'])[0]
test = pd.get_dummies(test, drop_first=True) 
test.shape


# In[62]:

X = np.concatenate((X_train,X_val),axis=0)
X = X.iloc[:, X.columns.isin(test.columns)]
Y = np.concatenate((Y_train,Y_val),axis=0)
X.shape


# Training Model with all train.csv datapoints:

# In[20]:

best_model = RandomForestRegressor(n_estimators = 51, random_state = 123)
best_model.fit(X,Y)


# Making Submission:

# In[64]:

#Need a list of ID's for below code:
predictions = np.expm1(best_model.predict(test))


# In[ ]:


with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))

