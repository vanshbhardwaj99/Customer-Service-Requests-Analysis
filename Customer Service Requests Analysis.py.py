#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1.Import a 311 NYC service request.

df=pd.read_csv(r'C:\Users\Vansham\Downloads\Data-Science-with-Python-Project-2--master\311_Service_Requests_from_2010_to_Present.csv')


# In[10]:


print(df.head())


# In[11]:


print(df.shape)


# In[12]:


print(df.isnull().sum())


# In[13]:


print(df[df['Closed Date'].isnull()])


# In[14]:


print(df.dtypes)


# In[15]:


import datetime as dt
import datetime, time

df['Created Date'] = pd.to_datetime(df['Created Date'])
print(df['Created Date'].dtype)


# In[16]:


df['Closed Date'] = pd.to_datetime(df['Closed Date'])
print(df['Closed Date'].dtype)


# In[17]:


df['Request_Closing_Time'] = df['Closed Date'] - df['Created Date']
df['Request_Closing_Time'].head()


# In[18]:


print(df['Complaint Type'].value_counts())


# In[19]:



df['Complaint Type'].value_counts().plot(kind="bar", color=list('rgbkymc'), figsize=(20,10))
plt.show()


# In[22]:


print(df['Status'].value_counts())

df['Status'].value_counts().plot(kind="barh", color=list('rgbkymc'), figsize=(20,10))
plt.show()


# In[23]:


print(df['City'].value_counts())

df['City'].value_counts().plot(kind="bar", color=list('rgbkymc'), figsize=(20,10))
plt.show()


# In[24]:


def toHour(timeDel):
    days = timeDel.days
    hours = round(timeDel.seconds/3600, 2)
    result = (days * 24) + hours
    return result


df['Request_Closing_In_Hour'] = df['Request_Closing_Time'].apply(toHour)
print(df['Request_Closing_In_Hour'].head())


# In[25]:


print(df['Request_Closing_In_Hour'].mean())


# In[26]:


df['Request_Closing_In_Hour'].value_counts().plot(kind='hist',bins=[0,2,4,6,8,10],rwidth=1)
plt.show()


# In[27]:


months = pd.Series({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
print(months)


# In[28]:


def getMonth(Date):
    a = str(Date)
    date = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
    return months[date.month]

df['Created_Month'] = df['Created Date'].apply(getMonth)
df['Created_Month']


# In[29]:


print(df['Created_Month'].value_counts())


# In[30]:


df['Created_Month'].value_counts().plot(kind="barh", color=list('rgbkymc'), figsize=(15,5))
plt.show()


# In[31]:


print(df['City'].isnull().sum())


# In[32]:


df['City'].fillna('NA', inplace=True)


# In[33]:


print(df['City'].head())


# In[34]:


grouped_df=df.groupby(['City', 'Complaint Type'])

RC_mean = grouped_df.mean()['Request_Closing_In_Hour']
print(RC_mean)


# In[35]:


print(RC_mean.isnull().sum())


# In[36]:


grouped_df = df.groupby(['City','Complaint Type']).agg({'Request_Closing_In_Hour': 'mean'})
print(grouped_df)


# In[37]:


print(grouped_df[grouped_df['Request_Closing_In_Hour'].isnull()])


# In[39]:


grouped_df=grouped_df.dropna()

print(grouped_df.isnull().sum())

print(grouped_df)


# In[40]:


g=grouped_df.sort_values(['City', 'Request_Closing_In_Hour'])
print(g)


# In[41]:


import scipy.stats as stats
from math import sqrt


# In[42]:


print(df['Complaint Type'].value_counts())

top_complaints_type = df['Complaint Type'].value_counts()[:5]
print(top_complaints_type)


# In[43]:


top_complaints_type_names = top_complaints_type.index
print(top_complaints_type_names)


# In[44]:


sample = df.loc[df['Complaint Type'].isin(top_complaints_type_names), ['Complaint Type', 'Request_Closing_In_Hour']]
print(sample.head())

print(sample.shape)

print(sample.isnull().sum())


# In[58]:


sample.dropna(inplace=True)
print(sample.isnull().sum())


# In[59]:


set1 = sample[sample['Complaint Type'] == top_complaints_type_names[1]].Request_Closing_In_Hour
print(set1.head())

set2 = sample[sample['Complaint Type'] == top_complaints_type_names[2]].Request_Closing_In_Hour
print(set2.head())

set3 = sample[sample['Complaint Type'] == top_complaints_type_names[3]].Request_Closing_In_Hour
print(set3.head())

set4 = sample[sample['Complaint Type'] == top_complaints_type_names[4]].Request_Closing_In_Hour
print(set4.head())

set5 = sample[sample['Complaint Type'] == top_complaints_type_names[0]].Request_Closing_In_Hour
print(set5.head())

f,pval=stats.f_oneway(set1, set2, set3, set4, set5)
print(f)
print(pval)


# In[61]:


if pval<0.05:
    print("Null Hypothesis is Rejected and All complain types average response time mean is not similar")
else:
    print("Null Hypothesis is Accepted and All complain types average response time mean is not similar")


# In[50]:


# Since, pvalue<0.05 we reject null hypothesis.
# Therefore, All Complain Types average response time mean is Not similar

# Try ChiSquare Test for part2  -  Are the type of complaint or service requested and location related?

# Null Hypothesis H0 : Complain Type and Location is not related
# Alternate Hypothesis H1 : Complain Type and Location is related

from scipy.stats import chi2_contingency
top_location = df['City'].value_counts()[:5]
print(top_location)


# In[51]:


top_location_names = top_location.index
print(top_location_names)


# In[52]:


sample2 = df.loc[(df['Complaint Type'].isin(top_complaints_type_names)) & (df['City'].isin(top_location_names)), ['Complaint Type', 'City']]
print(sample2.head())


# In[53]:


C_table=pd.crosstab(sample2['Complaint Type'], sample2['City'])
print(C_table)


# In[54]:


ch2, p, dof, tb1 = chi2_contingency(C_table)

print(ch2,p,dof)

# Since, p<0.05 we reject null hypothesis.
# Therefore, Complain Type and Location is related


# In[62]:


if p<0.05:
    print("Null Hypothesis is Rejected and All complain types average response time mean is not similar")
else:
    print("Null Hypothesis is Accepted and All complain types average response time mean is not similar")


# In[ ]:




