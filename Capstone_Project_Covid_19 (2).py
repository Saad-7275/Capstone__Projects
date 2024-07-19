#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PROJECT - COVID 19 TREND ANALYSIS

# Given the data about covid 19 patients we have to write code to visualize the impact and analyze the trend of the rate of infection and recovery as well as make predictions.
# The predictions will be made about the number of cases expected in a week in future based on the current trends



# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv("covid_19_clean_complete.csv")
data


# In[4]:


data.info()


# In[5]:


data.describe(include='all')


# In[6]:


data.columns


# In[7]:


data.rename(columns ={'Province/State': 'State','Country/Region':'Country'}, inplace = True)


# In[8]:


data.columns


# In[9]:


data.head(5)


# In[10]:


#cases on the last day
top = data[data['Date']== data['Date'].max()]


# In[11]:


top


# In[12]:


top.shape


# In[13]:


top['Date'].nunique()


# In[14]:


c = top.groupby('Country')['Confirmed','Deaths','Recovered','Active'].size().reset_index()


# In[15]:


gp = data.groupby('Country')[['Confirmed','Active','Recovered','Deaths']].sum().reset_index()


# In[16]:


gp


# In[17]:


pip install plotly


# In[18]:


import plotly.express as px
import matplotlib.pyplot as plt


# In[19]:


#world map --> Deaths
x = px.choropleth(gp,locations= 'Country', locationmode = 'country names', color = 'Deaths',
                 hover_name = 'Country', range_color = [1,40000], color_continuous_scale='Peach',
                 title = 'Deaths case country wise')
x.show()


# In[20]:


x = px.choropleth(gp,locations= 'Country', locationmode = 'country names', color = 'Deaths',
                 hover_name = 'Country', range_color = [1,50000], color_continuous_scale='Viridis',
                 title = 'Deaths case country wise')
x.show()


# In[21]:


x_active = px.choropleth(gp,locations= 'Country', locationmode = 'country names', color = 'Active',
                 hover_name = 'Country', range_color = [1,50000], color_continuous_scale='Viridis',
                 title = 'Active case country wise')
x_active.show()


# In[22]:


x = px.choropleth(gp,locations= 'Country', locationmode = 'country names', color = 'Confirmed',
                 hover_name = 'Country', range_color = [1,50000], color_continuous_scale='Viridis',
                 title = 'Confirmed case country wise')
x.show()


# In[23]:


x = px.choropleth(gp,locations= 'Country', locationmode = 'country names', color = 'Recovered',
                 hover_name = 'Country', range_color = [1,50000], color_continuous_scale='Viridis',
                 title = 'Recovered case country wise')
x.show()


# In[24]:


data.head(5)


# In[25]:


data.info()


# In[26]:


data['Date'] = pd.to_datetime(data['Date'])
#YYYY-MM-DD
data['Date'] = data['Date'].dt.date


# In[27]:


data['Date']


# In[28]:


data.head()


# In[29]:


# animation_frame  = 'Date'
x = px.choropleth(data,locations= 'Country', 
                  locationmode = 'country names', 
                  color = 'Deaths',
                 hover_name = 'Country', 
                  animation_frame  = 'Date',
                  range_color = [1,50000], 
                  color_continuous_scale='Viridis',
                 title = 'Deaths case country wise')
x.show()


# In[30]:


t_cases = data.groupby('Date')['Confirmed'].sum().reset_index()
t_cases


# In[31]:


t_cases_active = data.groupby('Date')['Active'].sum().reset_index()
t_cases_active


# In[32]:


t_cases_death = data.groupby('Date')['Deaths'].sum().reset_index()
t_cases_death


# In[33]:


t_cases_recovery = data.groupby('Date')['Recovered'].sum().reset_index()
t_cases_recovery


# Group data by date and sum the values for relevant columns
t_cases = data.groupby('Date')['Confirmed'].sum().reset_index()
t_cases_active = data.groupby('Date')['Active'].sum().reset_index()
t_cases_death = data.groupby('Date')['Deaths'].sum().reset_index()
t_cases_recovery = data.groupby('Date')['Recovered'].sum().reset_index()


#plot date vs confirmed cases using barplot 
plt.figure(figsize=(14, 7))
plt.bar(t_cases['Date'], t_cases['Confirmed'], color='red')
plt.xlabel('Date')
plt.ylabel('Confirmed cases')
plt.title('Total confirmed cases over time')
plt.xticks(rotation=45)
plt.show()





#plot date vs deaths cases using barplot 
plt.figure(figsize=(14, 7))
plt.bar(t_cases_death['Date'], t_cases_death['Deaths'], color='blue')
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.title('Total deaths over time')
plt.xticks(rotation=45)
plt.show()




#top 20 countries with most cases
t_20 = gp.groupby('Country')['Active'].sum().reset_index().sort_values(by = 'Active', ascending = False).head(20)
t_20





# Bar plot using Seaborn for top 20 countries with the most active cases
plt.figure(figsize=(14, 7))
sns.barplot(y=t_20.Active, x=t_20.Country, palette='viridis')
plt.xlabel('Country')
plt.ylabel('Active Cases')
plt.title('Top 20 Countries with Most Active Cases')
plt.xticks(rotation=45)
plt.show()





##Facebook Prophet
##--> an open sourve tool for forecasting the time series 




t_cases #confirmed
t_cases_active #active
t_cases_death #death
t_cases_recovery #recovery



from prophet import Prophet



t_cases.head()



# t_cases --> comfirmed cases data
t_cases.columns = ['ds','y']
t_cases.head()


t_cases['ds'] = pd.to_datetime(t_cases['ds'])
t_cases


# In[45]:


t_cases.info()


# In[46]:


#create the model
model = Prophet()


# In[47]:


model.fit(t_cases)


# In[48]:


future = model.make_future_dataframe(periods = 7, freq = 'D')
future


# In[49]:


forecast = model.predict(future)
forecast


# In[50]:


forecast.info()


# In[51]:


forecast[['yhat','yhat_lower','yhat_upper']] = forecast[['yhat','yhat_lower','yhat_upper']].astype(int)


# In[52]:


forecast.info()


# In[53]:


forecast[['ds','yhat','yhat_lower','yhat_upper']]


# In[54]:


plot_time = model.plot(forecast)
plot_time


# In[55]:


model.plot_components(forecast)


# In[ ]:




