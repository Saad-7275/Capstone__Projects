#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
# You are the Data Scientist at a telecom company “Neo” whose customers are churning out to
# its competitors. You have to analyse the data of your company and find insights and stop your
# customers from churning out to other telecom companies.

# # A) Data Manipulation:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('customer_churn.csv')
df.head()


# In[3]:


# to get the overview of the dataset
df.info()


# In[4]:


# a. Extract the 5th column & store it in ‘customer_5’


# In[5]:


customer_5=df.iloc[:,4]
customer_5.head()


# In[6]:


c=df.loc[:,['Dependents']]
c.head()


# In[ ]:


# b. Extract the 15th column & store it in ‘customer_15’


# In[7]:


customer_15=df.iloc[:,14]
customer_15.head()


# In[8]:


a=df.loc[:,['StreamingMovies']]
a.head()


# In[ ]:


# c. Extract all the male senior citizens whose Payment Method is Electronic check &
#    store the result in ‘senior_male_electronic’


# In[10]:


senior_male_electronic=df[(df['gender']=='Male') &  (df['SeniorCitizen']==1) & (df['PaymentMethod']=='Electronic check')]
senior_male_electronic.shape


# In[ ]:


# d. Extract all those customers whose tenure is greater than 70 months or their
#    Monthly charges are more than 100$ & store the result in ‘customer_total_tenure’


# In[11]:


customer_total_tenure=df[(df['tenure']>70) | (df['MonthlyCharges']>100)]
customer_total_tenure


# In[ ]:


#  e. Extract all the customers whose Contract is of two years, payment method is Mailed
#     check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’


# In[12]:


two_mail_yes=df[(df['Contract']=="Two year") & (df['PaymentMethod']=='Mailed check') & (df['Churn']=='Yes')]
two_mail_yes


# In[ ]:


# f. Extract 333 random records from the customer_churn dataframe & store the result in
#    ‘customer_333’


# In[15]:


customer_333=df.sample(n=333)
customer_333


# In[ ]:


# g. Get the count of different levels from the ‘Churn’ column


# In[16]:


df['Churn'].value_counts()


# In[17]:


df['Contract'].value_counts()


# In[18]:


df['InternetService'].value_counts()


# # B) Data Visualization:

# In[ ]:


# a. Build a bar-plot for the ’InternetService’ column:
#     i. Set x-axis label to ‘Categories of Internet Service’
#     ii. Set y-axis label to ‘Count of Categories’
#     iii. Set the title of plot to be ‘Distribution of Internet Service’
#     iv. Set the color of the bars to be ‘orange’


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
x=df['InternetService'].value_counts().keys().tolist()
y=df['InternetService'].value_counts().tolist()


# In[19]:


df['InternetService'].value_counts().keys()


# In[21]:


plt.bar(x,y, color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')


# In[ ]:


# b. Build a histogram for the ‘tenure’ column:
#   i. Set the number of bins to be 30
#   ii. Set the color of the bins to be ‘green’
#   iii. Assign the title ‘Distribution of tenure’


# In[24]:


plt.hist(df['tenure'], color='green', bins=30)
plt.title('Distribution of Tenure')


# In[ ]:


# c. Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to
#    the y-axis & ‘tenure’ to the ‘x-axis’:
#     i. Assign the points a color of ‘brown’
#     ii. Set the x-axis label to ‘Tenure of customer’
#     iii. Set the y-axis label to ‘Monthly Charges of customer’
#     iv. Set the title to ‘Tenure vs Monthly Charges’


# In[26]:


plt.scatter(x=df['tenure'].head(100), y=df['MonthlyCharges'].head(100), color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()


# In[ ]:


# d. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis &
#     ‘Contract’ on the x-axis.


# In[27]:


df.boxplot(column='tenure', by=['Contract'], figsize=(10,5))


# # MODEL BUILDING

# # C) Linear Regression:

# In[ ]:


# a. Build a simple linear model where dependent variable is ‘MonthlyCharges’ and
#    independent variable is ‘tenure’
#     i. Divide the dataset into train and test sets in 70:30 ratio.
#     ii. Build the model on train set and predict the values on test set
#     iii. After predicting the values, find the root mean square error
#     iv. Find out the error in prediction & store the result in ‘error’
#     v. Find the root mean square error


# In[28]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[29]:


x=df.loc[:,['tenure']].values
y=df.loc[:,['MonthlyCharges']].values


# In[39]:


x_train, x_test , y_train , y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[40]:


x_test


# In[41]:


from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(x_train,y_train)


# In[42]:


y_pred = simpleLinearRegression.predict(x_test)
y_pred


# In[43]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse


# # D) Logistic Regression:

# In[ ]:


# a. Build a simple logistic regression modelwhere dependent variable is ‘Churn’ &
#    independent variable is ‘MonthlyCharges’
#      i. Divide the dataset in 65:35 ratio
#      ii. Build the model on train set and predict the values on test set
#      iii. Build the confusion matrix and get the accuracy score


# In[48]:


x = df.loc[:,['MonthlyCharges']].values
y = df.loc[:,['Churn']].values


# In[49]:


x_train, x_test , y_train , y_test=train_test_split(x,y,train_size=0.65,random_state=0)


# In[50]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[51]:


y_pred = logmodel.predict(x_test)
y_pred


# In[52]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test)


# In[ ]:


# b. Build a multiple logistic regression model where dependent variable is ‘Churn’ &
#    independent variables are ‘tenure’ & ‘MonthlyCharges’
#      i. Divide the dataset in 80:20 ratio
#      ii. Build the model on train set and predict the values on test set
#      iii. Build the confusion matrix and get the accuracy score


# In[53]:


x = df.loc[:,['MonthlyCharges','tenure']].values
y = df.loc[:,['Churn']].values


# In[54]:


x_train, x_test , y_train , y_test=train_test_split(x,y,train_size=0.80,random_state=0)


# In[55]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[56]:


y_pred = logmodel.predict(x_test)
y_pred


# In[57]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test)


# # E) Decision Tree:

# In[ ]:


# a. Build a decision tree model where dependent variable is ‘Churn’ & independent
#    variable is ‘tenure’
#      i. Divide the dataset in 80:20 ratio
#      ii. Build the model on train set and predict the values on test set
#      iii. Build the confusion matrix and calculate the accuracy


# In[58]:


x = df.loc[:,['tenure']].values
y = df.loc[:,['Churn']].values


# In[59]:


x_train, x_test , y_train , y_test=train_test_split(x,y,test_size=0.20)


# In[60]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)


# In[61]:


y_pred = classifier.predict(x_test)
y_pred


# In[62]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# # F) Random Forest:
# 

# In[ ]:


# a. Build a Random Forest model where dependent variable is ‘Churn’ & independent
#    variables are ‘tenure’ and ‘MonthlyCharges’
#     i. Divide the dataset in 70:30 ratio
#     ii. Build the model on train set and predict the values on test set
#     iii. Build the confusion matrix and calculate the accuracy


# In[63]:


x = df.loc[:,['tenure','MonthlyCharges']].values
y = df.loc[:,['Churn']].values


# In[64]:


x_train, x_test , y_train , y_test=train_test_split(x,y,test_size=0.30)


# In[65]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)


# In[66]:


y_pred = clf.predict(x_test)
y_pred


# In[67]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:




