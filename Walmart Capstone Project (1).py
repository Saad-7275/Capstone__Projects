#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Statistical Analysis, EDA and outlier Analysis, handling the Missing Values

# In[104]:


df_new = pd.read_csv('Walmart Dataset.csv')

df_new.tail(30)


# In[89]:


df_new.reset_index(inplace = True)

df_new['Date'] = pd.to_datetime(df_new['Date'])

df_new.set_index('Date', inplace = True)


# In[90]:


df_new.head(10)


# In[91]:


df_new.info()


# In[92]:


df_new.isnull().sum().sum()


# In[93]:


df_new.describe().T


# In[94]:


df_new.columns


# In[95]:


col = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

correlation = df_new[col].corr()

correlation


# In[96]:


df_new.head(10)


# In[109]:


col_1 = ['Weekly_Sales','Store']

sort_col_1 = df_new[col_1].sort_values(by = 'Weekly_Sales', ascending = True)


# In[110]:


sort_col_1.head(10)


# In[ ]:





# In[112]:


import seaborn as sns


# In[117]:


sns.heatmap(correlation, cmap = 'coolwarm', annot = True, xticklabels = 'auto', yticklabels = 'auto' )


# #### Ploting the Box Plot for Outlier Detection

# In[118]:


df.columns


# In[128]:


col_2 = ['Weekly_Sales','Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

for col in col_2:
    sns.set()
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel('Values')
    plt.show()
    


# In[ ]:


### Hence there are no major OUtliers Present in the Dataset Provided.


# In[101]:


# a.) Therefore from the COrrelation data we have found that the Unemployment rate does affect the
# Weekly_Sales , and the most affected Store is 33


# In[ ]:





# In[105]:


## b.) The Reason behind Weekly_Sales Showing Seasonal Trends is beacuse of the Holiday Flag being "0" or "1"


# In[106]:


## c.) Temperature has a negative correlation with the weekly sales therefore it does affect the weekly sales inversly.


# In[ ]:


##D.) Consumer Sale Price(CPI) is affecting the Weekly Sales inversely with negative correlation Between the TWo.


# In[ ]:


# E.) The Top Performing store is 14 with a weekly sale of 3818686.45.


# In[ ]:


# F.) The Worst Performing Store is 33 with a weekly sale of 209986.25

# The DIfference between the Highest and Worst Performing STores is 3608700.2 (weekly_sales difference)


# ### Predictive Modeling To Forecast the Sales

# In[51]:


df = pd.read_csv('Walmart Dataset.csv')
df.set_index('Date', inplace=True)
# There are about 45 different stores in this dataset. Lets select the any store id from 1-45
a= int(input("Enter the store id:"))
store = df[df.Store == a]
sales = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales.dtypes


# In[3]:


sales.head(10)


# In[5]:


#remove date from index to change its dtype because it clearly isnt acceptable.
sales.reset_index(inplace = True)

#converting 'date' column to a datetime type
sales['Date'] = pd.to_datetime(sales['Date'])
# resetting date back to the index
sales.set_index('Date',inplace = True)


# In[6]:


sales.Weekly_Sales.plot(figsize=(10,6), title= 'Weekly Sales of a Store', fontsize=14, color = 'blue')
plt.show()


# In[14]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(sales.Weekly_Sales, freq=7, extrapolate_trend = 'freq')  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(12, 10)
plt.show()


# In[15]:


#lets compare the 2012 data of two stores
# Lets take store 5 data for analysis

store_5 = df[df.Store == 5]

# there are about 45 different stores in this dataset.

sales_5 = pd.DataFrame(store_5.Weekly_Sales.groupby(store_5.index).sum())
sales_5.dtypes

# Grouped weekly sales by store 6

#remove date from index to change its dtype because it clearly isnt acceptable.
sales_5.reset_index(inplace = True)

#converting 'date' column to a datetime type

sales_5['Date'] = pd.to_datetime(sales_5['Date'])

# resetting date back to the index

sales_5.set_index('Date',inplace = True)


# In[43]:


y1 = sales.Weekly_Sales
y2 = sales_5.Weekly_Sales


# In[17]:


y1['2012'].plot(figsize=(15, 6),legend=True, color = 'chocolate')
y2['2012'].plot(figsize=(15, 6), legend=True, color = 'turquoise')
plt.ylabel('Weekly Sales')
plt.title('Store4 vs Store5 on 2012', fontsize = '16')
plt.show()


# In[18]:


# Clearly we can see the irregularities 


# In[19]:


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 5)

import itertools

# Generate all different combinations of p, d and q triplets

pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, d and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]


# In[20]:


import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(y1,
                                order=(4, 4, 3),
                                seasonal_order=(1, 1, 0, 52),   #enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# In[21]:


plt.style.use('seaborn-pastel')
results.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[22]:


pred = results.get_prediction(start=pd.to_datetime('2012-07-27'), dynamic=False)
pred_ci = pred.conf_int()


# In[23]:


ax = y1['2010':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Time Period')
ax.set_ylabel('Sales')
plt.legend()

plt.show()


# In[24]:


y_forecasted = pred.predicted_mean
y_truth = y1['2012-7-27':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[25]:


pred_dynamic = results.get_prediction(start=pd.to_datetime('2012-7-27'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


# In[26]:


ax = y1['2010':].plot(label='observed', figsize=(12, 8))

pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2012-7-26'), y1.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Time Period')
ax.set_ylabel('Sales')

plt.legend()
plt.show()



# In[27]:


import numpy as np

# Extract the predicted and true values of our time series

y_forecasted = pred_dynamic.predicted_mean
print(y_forecasted)


# In[28]:


y_truth = y1['2012-7-27':]

print(y_truth)


# In[29]:


# Compute the Root mean square error
rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))


# In[30]:


Residual= y_forecasted - y_truth
print("Residual for Store1",np.abs(Residual).sum())


# In[31]:


# Get forecast 12 weeks ahead in future
pred_uc = results.get_forecast(steps=12)

print(pred_uc)


# In[32]:


# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()


# In[ ]:


ax = y1.plot(label='observed', figsize=(12, 8))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Time Period')
ax.set_ylabel('Sales')

plt.legend()
plt.show()



# In[ ]:




