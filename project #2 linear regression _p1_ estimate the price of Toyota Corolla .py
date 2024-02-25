#!/usr/bin/env python
# coding: utf-8

# # LinearRegression

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
from sklearn.model_selection import train_test_split


# ###### loading dataset and spliring into test and train and validatook set

# In[3]:


data = pd.read_excel("data/ToyotaCorolla.xls","data")
y = data[data.columns[2]]
x = data[data.columns[3:]]
x = pd.get_dummies(x) #change non numerical attributes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3 ,random_state=42 )
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


# ## Ordinary Least Squares

# ###### LinearRegression will take in its fit method arrays x, y and will store the coefficients  of the linear model in its coef_ member:

# In[410]:


# object
reg1 = linear_model.LinearRegression()
# Train the model
reg1.fit(x_train, y_train)
# predictions
y_pred_train = reg1.predict(x_train)
y_pred_val = reg1.predict(x_val)
y_pred_test1 = reg1.predict(x_test)
# The coefficients
print('Coefficients: \n', reg1.coef_)
# error
msev1=mean_squared_error(y_val, y_pred_val)
maev1=mean_absolute_error(y_val, y_pred_val)
mset1=mean_squared_error(y_test, y_pred_test1)
maet1=mean_absolute_error(y_test, y_pred_test1)
print('Mean squared error validation set: %.2f' % msev1)
print('Mean_absolute_error validation set: %.2f' % maev1 )
print('Mean squared error test: %.2f' % mset1)
print('Mean_absolute_error test: %.2f' % maet1)
# The coefficient of determination: 1 is perfect prediction
r2_train1=r2_score(y_train, y_pred_train)
r2_v1=r2_score(y_val, y_pred_val)
r2_test1=r2_score(y_test, y_pred_test1)
print( 'R2 score train: %.2f' %r2_train1)
print( 'R2 score validation set: %.2f' %r2_v1)
print( 'R2 score test: %.2f' %r2_test1)
# Plot outputs
fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
fig.set_figheight(4)
fig.set_figwidth(15)
ax1.scatter(y_pred_train, y_train,  color='#FF00FF')
ax1.plot(y_train, y_train ,color='darkblue')
ax1.set_title('train')

ax2.scatter(y_pred_val, y_val,  color='#A500A5')
ax2.plot(y_val, y_val, color='darkblue')
ax2.set_title('validation')

ax3.scatter(y_pred_test1, y_test,  color='#660066')
ax3.plot(y_test, y_test, color='darkblue')
ax3.set_title('test')
axs=(ax1,ax2,ax3)
for ax in axs:
    ax.set(xlabel='predicted price', ylabel='real price')


# ## Ridge regression

# In[411]:


# object
reg2 = linear_model.Ridge(alpha=10)
# Train the model
reg2.fit(x_train, y_train)
# predictions
y_pred_train = reg2.predict(x_train)
y_pred_val = reg2.predict(x_val)
y_pred_test2 = reg2.predict(x_test)
# The coefficients
print('Coefficients: \n', reg2.coef_)
# The mean squared error
msev2=mean_squared_error(y_val, y_pred_val)
maev2=mean_absolute_error(y_val, y_pred_val)
mset2=mean_squared_error(y_test, y_pred_test2)
maet2=mean_absolute_error(y_test, y_pred_test2)
print('Mean squared error validation set: %.2f' % msev2)
print('Mean_absolute_error validation set: %.2f' % maev2 )
print('Mean squared error test: %.2f' % mset2)
print('Mean_absolute_error test: %.2f' % maet2)
# The coefficient of determination: 1 is perfect prediction
r2_train2=r2_score(y_train, y_pred_train)
r2_v2=r2_score(y_val, y_pred_val)
r2_test2=r2_score(y_test, y_pred_test2)
print( 'R2 score train: %.2f' %r2_train2)
print( 'R2 score validation set: %.2f' %r2_v2)
print( 'R2 score test: %.2f' %r2_test2)
# Plot outputs
fig, (ax1,ax2,ax3_2) = plt.subplots(1, 3)
fig.set_figheight(4)
fig.set_figwidth(15)
ax1.scatter(y_pred_train, y_train,  color='#0091FF')
ax1.plot(y_train, y_train ,color='darkblue')
ax1.set_title('train')

ax2.scatter(y_pred_val, y_val,  color='#006BBD')
ax2.plot(y_val, y_val, color='darkblue')
ax2.set_title('validation')

ax3_2.scatter(y_pred_test2, y_test,  color='#00365F')
ax3_2.plot(y_test, y_test, color='darkblue')
ax3_2.set_title('test')
axs=(ax1,ax2,ax3_2)
for ax in axs:
    ax.set(xlabel='predicted price', ylabel='real price')


# ## Lasso regression

# In[412]:


# object
reg3 = linear_model.Lasso(alpha=8)
# Train the model
reg3.fit(x_train, y_train)
# predictions
y_pred_train = reg3.predict(x_train)
y_pred_val = reg3.predict(x_val)
y_pred_test3 = reg3.predict(x_test)
# The coefficients
print('Coefficients: \n', reg3.coef_)
# The mean squared error
msev3=mean_squared_error(y_val, y_pred_val)
maev3=mean_absolute_error(y_val, y_pred_val)
mset3=mean_squared_error(y_test, y_pred_test3)
maet3=mean_absolute_error(y_test, y_pred_test3)
print('Mean squared error validation set: %.2f' % msev3)
print('Mean_absolute_error validation set: %.2f' % maev3 )
print('Mean squared error test: %.2f' % mset3)
print('Mean_absolute_error test: %.2f' % maet3)
# The coefficient of determination: 1 is perfect prediction
r2_train3 = r2_score(y_train, y_pred_train)
r2_v3 = r2_score(y_val, y_pred_val)
r2_test3 = r2_score(y_test, y_pred_test3)
print( 'R2 score train: %.2f' %r2_train3)
print( 'R2 score validation set: %.2f' %r2_v3)
print( 'R2 score test: %.2f' %r2_test3)
# Plot outputs
fig, (ax1,ax2,ax3_3) = plt.subplots(1, 3)
fig.set_figheight(4)
fig.set_figwidth(15)
ax1.scatter(y_pred_train, y_train,  color='#00FF00')
ax1.plot(y_train, y_train ,color='darkblue')
ax1.set_title('train')

ax2.scatter(y_pred_val, y_val,  color='#02B602')
ax2.plot(y_val, y_val, color='darkblue')
ax2.set_title('validation')

ax3_3.scatter(y_pred_test3, y_test,  color='#005700')
ax3_3.plot(y_test, y_test, color='darkblue')
ax3_3.set_title('test')
axs=(ax1,ax2,ax3_3)
for ax in axs:
    ax.set(xlabel='predicted price', ylabel='real price')


# ## comparing methods

# In[413]:



d = {'MSE V': [msev1, msev2,msev3]
     ,'MSE T': [mset1,mset2,mset3]
     ,'MAE V': [maev1, maev2,maev3]
     ,'MAE T': [maet1, maet2,maet3]
     ,'R2 V': [r2_v1, r2_v2,r2_v3]
     ,'R2 T': [r2_test1,r2_test2,r2_test3]}

df = pd.DataFrame(data=d , index=['Least Squares Regression','Ridge Regression','Lasso Regression'])

fig, (ax3_1,ax3_2,ax3_3) = plt.subplots(1, 3)
fig.set_figheight(4)
fig.set_figwidth(15)
ax3_1.scatter(y_pred_test1, y_test,  color='#660066')
ax3_1.plot(y_test, y_test, color='darkblue')
ax3_1.set_title('Ordinary Least Squares')
ax3_2.scatter(y_pred_test2, y_test,  color='#00365F')
ax3_2.plot(y_test, y_test, color='darkblue')
ax3_2.set_title('Ridge')
ax3_3.scatter(y_pred_test3, y_test,  color='#005700')
ax3_3.plot(y_test, y_test, color='darkblue')
ax3_3.set_title('Lasso')
axs=(ax3_1,ax3_2,ax3_3)
for ax in axs:
    ax.set(xlabel='predicted price', ylabel='real price')
df


# In[ ]:




