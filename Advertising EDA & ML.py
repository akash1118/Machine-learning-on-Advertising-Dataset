#!/usr/bin/env python
# coding: utf-8

# In[158]:


# importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[159]:


#reading the Advertising dataset
train = pd.read_csv('Advertising (1).csv')


# In[160]:


#checking the first five rows of the advertising dataset
train.head()


# In[161]:


#checking the number of rows and columns of the dataset
train.shape


# In[162]:


#checking the null values
train.isnull().sum()


# In[163]:


#checking the null values as well as the datatypes of the dataset
train.info()


# From the given dataset it is clearly visible that there are no missing values. It is a regression problem.

# In[164]:


#Generating statistical info i.e., count, mean, quartiles, max
train.describe()


# In[165]:


#Plotting missing values
sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')


# No missing values found

# In[223]:


# Plotting the histogram to check the type of distribution
sns.displot(train['Sales'],kde=False,color='darkred', bins=20)


# It seems like a normal distribution

# In[167]:


#Checking the columns of the dataset
train.columns


# In[168]:


#creating the variables features
features = ['Unnamed: 0', 'TV', 'Radio', 'Newspaper', 'Sales']


# In[169]:


#displaying features
train[features]


# In[170]:


#creating a duplicate dataset train1
train1 = train.copy()
train1 = train.drop(['Unnamed: 0'], axis = 1)


# In[171]:


#checking the copied dataset
train1.head()


# In[172]:


# Multivariate analysis

sns.pairplot(train1, x_vars = train1.iloc[:, :], y_vars= train1.iloc[:, :], kind = 'scatter', diag_kind = 'kde')


# From the pairplot it can be observed that TV plays a major role in increase of sales.

# In[173]:


##plotting boxplot in order to get the outliers present in the dataset
fig, axs = plt.subplots(3, figsize = (2,7))
plt1 = sns.boxplot(y= train1['TV'], orient = "v", ax = axs[0])
plt2 = sns.boxplot(y= train1['Newspaper'],orient = "v", ax = axs[1])
plt3 = sns.boxplot(y = train1['Radio'], orient = "v", ax = axs[2])
plt.tight_layout()


# In[174]:


#Describing the dataset again to validate the percentiles with the boxplot
train1.describe()


# In[175]:


# Univariate analysis
sns.boxplot(x = train1['Sales'])


# In[176]:


# Correlation between variables
sns.heatmap(train1.corr(), cmap ='viridis' , annot =True)


# As is visible from the pairplot and the heatmap, the variable TV seems to be most correlated with Sales. So let's go ahead and perform simple linear regression using TV as our feature variable.

# In[180]:


#dropping the sales columns from the independent columns data
X= train1.drop('Sales', axis=1)
y= train1[["Sales"]]


# In[181]:


#Checking the first five rows of the independent features
X.head(10)


# In[182]:


#checking the first few rows of the dependent feature
y.head()


# In[183]:


#Appling linear regression on the whole dataset using sklearn
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()


# In[196]:


#fitting the model
model = lr.fit(X, y)


# In[197]:


#getting the intercept of the model
model.intercept_


# In[198]:


#getting the coefficients of the model
model.coef_


# In[199]:


#predicting the model 
model.predict(X)


# In[200]:


#impoting class mean_squared_error to calculate the mean squared error
from sklearn.metrics import mean_squared_error


# In[201]:


#calculating the mean squared error(MSE)
MSE= mean_squared_error(y, model.predict(X))


# In[202]:


#calculating the root mean squared error(RMSE)
RMSE = np.sqrt(MSE)


# In[203]:


RMSE


# In[ ]:





# In[189]:


#installing the library statsmodel
get_ipython().system('pip install statsmodels')


# In[190]:


#importing statsmodels library to perform linear regression using statsmodels library
import statsmodels.api as sm


# In[206]:


#creating the model
lm= sm.OLS(y, X)


# In[207]:


#fitting the model
model= lm.fit()


# In[208]:


#getting the summary of the model
model.summary()


# In[209]:


#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 1)


# In[210]:


#checking first five rows of the independent features of the training set
X_train.head()


# In[211]:


#checking first five rows of the independent features of the training set
y_train.head()


# In[212]:


#appling linear regression the the training set using sklearn
lm= LinearRegression()
model= lm.fit(X_train, y_train)


# In[214]:


#predicting X_train and calculating the root mean squared error(RMSE) on the training set
predictions = model.predict(X_train)
RMSE = np.sqrt(mean_squared_error(y_train, predictions))


# In[215]:


RMSE


# In[217]:


#Predicting X_test
predictions1 = model.predict(X_test)


# In[218]:


#calculating the root mean squared error(RMSE) on the test set
RMSE = np.sqrt(mean_squared_error(y_test, predictions1 ))


# In[219]:



RMSE


# In[220]:


#checking the accuracy of the model
r2_score = model.score(X_test,y_test)
print(r2_score*100,'%')


# In[ ]:





# In[ ]:





# In[ ]:




