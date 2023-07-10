#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt


# In[2]:


data1 = pd.read_csv("Downloads/Video_Game_Sales.csv")
data1


# In[3]:


data1.shape


# In[4]:


data1.columns


# In[5]:


data1.index


# In[6]:


data1.values


# In[7]:


data1.describe()


# In[8]:


data1.isnull().sum()


# In[9]:


data1.columns[data1.isna().any()]


# In[10]:


#data1.fillna(method="ffill")


# In[3]:


data1.fillna(method="bfill",inplace=True)
data1


# In[12]:


data1.isnull().sum()


# In[13]:


data1


# In[4]:


data1.fillna(method="ffill",inplace=True)
data1


# In[5]:


data1.isnull().sum()


# In[16]:


#data1.[["Critic_Score","Critic_Count","User_Score","User_Count","Rating"]].fillna(mean(),inplace=True)


# In[17]:


#cols=data1.loc[:,["Critic_Score","Critic_Count","User_Score","User_Count","Rating"]
#cols            


# In[18]:


#cols.fillna(mean(),inplace=True)


# In[19]:


#data1.interpolate()


# In[20]:


# from sklearn.impute import SimpleImputer 


# In[21]:


#cols=data1.loc[:,["Year_of_Release", "Publisher", "Critic_Score", "Critic_Count",
  #     "User_Score", "User_Count", "Rating"]]
#cols


# In[22]:


#my_imputer= SimpleImputer()


# In[23]:


#imputed_values=my_imputer.fit_transform(cols)
#imputed_values


# In[24]:


#new_values=pd.DataFrame(imputed_values)
#new_values


# In[25]:


#new_values.columns=cols.columns
#new_values


# In[26]:


# data1.score=data1.score.str.replace("s","")


# In[27]:


#catigorical data 
data1.Rating.unique()


# In[28]:


data1.Rating.map({'E':0, 'M':1, 'T':2, 'E10+':3, 'K-A':4, 'AO':5, 'EC':6, 'RP':7})


# In[10]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[11]:


le=LabelEncoder()


# In[12]:


data1.Rating=le.fit_transform(data1.Rating)


# In[13]:


data1


# In[33]:


#data1.unique()


# In[14]:


oh=OneHotEncoder(sparse=False)


# In[17]:


oh_values=pd.DataFrame(oh.fit_transform(data1[["Name"]]))
oh_values


# In[18]:


data1.drop("Name",axis=1,inplace=True)


# In[16]:


new_data=pd.concat([oh_values,data1],axis=1)


# new_data=pd.concat([data1,oh_values,axis=1)

# In[38]:


new_data


# In[40]:


#data visualization
plt.plot(data1.NA_Sales,data1.EU_Sales)


# In[41]:


data1.Critic_Score.hist()


# In[ ]:


data1.describe()


# In[6]:


data1.hist()


# In[20]:


plt.barh(data1.NA_Sales,data1.EU_Sales,data1.JP_Sales)


# In[21]:


data1.plot(kind="hist")


# In[ ]:


data1.plot(kind="bar")


# In[ ]:


data1.plot(kind="line")


# In[ ]:


#linear regression
x=data1.loc[:13900,["Global_Sales"]]
y=data1.loc[:13900,["Rating"]]
x_test=data.loc[13900:,["Global_Sales"]]
y_desired=data.loc[13900:,["Rating"]]


# In[ ]:


plt.scatter(x,y)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression()


# In[ ]:


model.fit(x,y)


# In[ ]:


y_predicted=model.predict(x_test)
y_predicted


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


import statsmodels.formula.api as sm


# In[ ]:


plt.scatter(x_test,y_predicted)


# In[ ]:


plt.scatter(x_test,y_desired)


# In[ ]:


import sklearn.metrics as mc


# In[ ]:


mc.mean_absolute_error(y_desired,y_predicted)


# In[ ]:


mc.mean_squared_error(y_desired,y_predicted)


# In[ ]:


z=data1.[["Global_Sales","Critic_Score"]]
y=data1.[["Rating"]]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


z_tain,z_test,y_train,y_test=train_test_split(z,y,test_size=.2)


# In[ ]:


z_tain
z_test
y_train
y_test


# In[ ]:


sns.pairplot(data)


# In[ ]:


sns.pairplot(data,z_vars=["Global_Sales","Critic_Score"],y_vars["Rating"],diag_kind=None,kind="scatter",size=5)


# In[ ]:


model.fit(z_tain,y_train)


# In[ ]:


y_predicted=model.predict(z_test)
y_predicted


# In[ ]:


y_test


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


lr=sm.ols(formula='')


# In[ ]:


plt.scatter(data1.Global_Sales,data1.Critic_Score)


# In[ ]:


sns.relplot(data1=dt,x='Global_Sales',y='Rating',kind="scatter",hue="Genre",col="Publisher")


# In[ ]:


x=data1.Global_Sales
y=data1.Critic_Score


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_tain,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[ ]:


plt.scatter(x,y)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr=LinearRegression


# In[ ]:


y_pedict=lr.predict(x_test)
y_pedict


# In[ ]:


plt.plot()


# In[ ]:


#polynomial regression


# In[1]:


from sklearn.preproccessing import PolynomialFeatures


# In[ ]:


polyF=PolynomialFeatures(degree=2)


# In[ ]:


x_poly=polyF.fit_transform(x)


# In[ ]:


model.fit(x_poly,y)


# In[ ]:


y_poly_predict=model.predict(x_poly)


# In[ ]:


plt.scatter(x)
plt.plot(x,y_poly_predict)


# In[ ]:


#pipeline
from sklearn.pipeline import PipeLine


# In[ ]:


inpt=[('polynomial',PolynomialFeatures(degree=2)),('model',LinearRegression())]


# In[ ]:


pip=PipeLine(inpt)


# In[ ]:


pip.fit(x,y)


# In[ ]:


y_pip_predict= pip.predict(x)
y_pip_predict


# In[ ]:


plt.scatter(x,y)
plt.plot(x,y_pip_predict,color="green")


# In[ ]:


#linear regression with statsmodels
lr=sm.ols(formula='Global_Sales','Critic_Score',data1=data).fit()


# In[ ]:


y_sm_prediction=lr.predict({"Global_Sales":4.55,"Critic_Score":85.2})
y_sm_prediction


# In[ ]:


d=lr.predict(x_test)
d


# In[ ]:


lr.params


# In[ ]:


lr,summary


# In[ ]:


#svm
from sklearn import svm


# In[ ]:


c3=svm.SVC(kernel='poly',degree=3)


# In[ ]:


c3.fit(x_train,y_train)


# In[ ]:


y_pred_svm=c3.predict(x_test)
y_pred_svm


# In[ ]:


import sklearn.metrics as mc
dir(mc)


# In[ ]:


conf=confusion_matrix(y_test,y_predict)#GNB


# In[ ]:


GNB.predict_proba(x_test)


# In[ ]:


report=mc.classification_report(y_test,y_svm_predict)#SVM
report


# In[ ]:


mc.accuracy_score(y_test,y_svm_predict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




