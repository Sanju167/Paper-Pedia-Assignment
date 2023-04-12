#!/usr/bin/env python
# coding: utf-8

# # SALES DATA ANALYSIS

# # Import Modules 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# -----------warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# # Q1 ANSWERS 

# ## DATA CLEANING

# In[2]:


df1=pd.read_csv(r"C:\Users\HP\Documents\company assignment\train.csv")
df1


# ## checked dimension

# In[3]:


df1.shape


# ### Checking null values

# In[4]:


df1.isna().sum()


# In[5]:


df1.info()


# In[6]:


df1["Product_Category_2"].value_counts()


# In[7]:


df1['Product_Category_2'].fillna(df1["Product_Category_2"].mean(),inplace=True)


# In[8]:


df1["Product_Category_3"].value_counts()


# In[9]:


df1['Product_Category_3'].fillna(df1["Product_Category_3"].mean(),inplace=True)


# In[10]:


df1['Product_Category_3'].unique()


# ## Checked Nosie

# In[11]:


df1.columns


# In[12]:


col_1=['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3', 'Purchase']


# In[13]:


for i in col_1:
    print(df1[i].unique(),"\n")


# In[14]:


df1["Gender"].value_counts()


# In[15]:


map_gender_1 = {'F':1, 'M':2}


# In[16]:


df1["Gender"]=df1["Gender"].map(map_gender_1).astype("int64")


# In[17]:


df1["Age"].value_counts()


# In[18]:


map_age_train = {'0-17': 1, '18-25': 5, '26-35': 7, '36-45': 6, '46-50': 4, '51-55': 3, '55+': 2}


# In[19]:


df1["Age"]=df1["Age"].map(map_age_train)


# In[20]:


df1["Stay_In_Current_City_Years"]=df1["Stay_In_Current_City_Years"].replace('\D','',regex=True)


# In[21]:


df1["Stay_In_Current_City_Years"].unique()


# In[22]:


df1["Stay_In_Current_City_Years"]=pd.to_numeric(df1["Stay_In_Current_City_Years"], errors="coerce")


# In[23]:


df1.head(3)


# In[24]:


df1.drop(columns="Product_ID",inplace=True)


# In[25]:


df1.info()


# # Test Dataset 

# In[26]:


df2=pd.read_csv(r"C:\Users\HP\Documents\company assignment\test.csv")
df2


# In[27]:


df2.shape


# In[28]:


df2.info()


# ## Checked Null

# In[29]:


df2.isna().sum()


# In[30]:


df2['Product_Category_2'].fillna(df2["Product_Category_2"].mean(),inplace=True)


# In[31]:


df2['Product_Category_3'].fillna(df2["Product_Category_3"].mean(),inplace=True)


# In[32]:


df2.columns


# ## Checked Noise

# In[33]:


col_2=['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3']


# In[34]:


for i in col_2:
    print(df2[i].unique(),"\n")


# In[35]:


df2["Gender"].value_counts()


# In[36]:


map_gender_2 = {'F':1, 'M':2}


# In[37]:


df2["Gender"]=df2["Gender"].map(map_gender_2)


# In[38]:


df2["Gender"]=pd.to_numeric(df2["Gender"], errors="coerce")


# In[39]:


df2["Age"].value_counts()


# In[40]:


map_age_test = {'0-17': 1, '18-25': 5, '26-35': 7, '36-45': 6, '46-50': 4, '51-55': 3, '55+': 2}


# In[41]:


df2["Age"]=df2["Age"].map(map_age_test)


# In[42]:


df2["Stay_In_Current_City_Years"]=df2["Stay_In_Current_City_Years"].replace('\D','',regex=True)


# In[43]:


df2["Stay_In_Current_City_Years"]=pd.to_numeric(df2["Stay_In_Current_City_Years"], errors="coerce")


# In[44]:


df2.drop(columns="Product_ID",inplace=True)


# In[45]:


df2.info()


# # DATA PREPROCESSING

# In[46]:


df1.describe()


# In[47]:


df1.corr()


# In[48]:


df1.head()


# In[49]:


Box_plot_1=['Age','Occupation','Stay_In_Current_City_Years',
            'Marital_Status','Product_Category_1','Product_Category_2']


# In[50]:


for c in Box_plot_1:
  percentile25 = df1[c].quantile(0.25)
  percentile75 = df1[c].quantile(0.75)
  IQR = percentile75 - percentile25
  Upperlimit = percentile75 + 1.5*IQR
  Lowerlimit = percentile25 - 1.5*IQR
  df1= df1[df1[c]<=Upperlimit]
  df1 = df1[df1[c]>=Lowerlimit]
  plt.figure()
  sns.boxplot(y=c, data =df1)


# ### Pairplot graph 

# In[131]:


df1_pp=df1[['Age','Occupation','Stay_In_Current_City_Years',
            'Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']]

df1_pp.head(2)


# In[132]:


# sns.pairplot(df1_pp, hue='Age')


# Pair plot takes lots of time so for easy convenience I have also attached the image of it 

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# ### HeatMap

# In[130]:


sns.heatmap(df1.corr(),annot = True)
sns.set(rc={'figure.figsize':(20,20)})


# # DATA VISUALIZATION

# ### Visualize an individual column

# In[127]:


df1["Age"].plot.hist(color='green')
sns.set(rc={'figure.figsize':(15,5)})


# In[55]:


sns.distplot(df1["Occupation"],color='Red')
sns.set(rc={'figure.figsize':(15,5)})


# In[56]:


df1["Stay_In_Current_City_Years"].plot.hist(color='blue')
sns.set(rc={'figure.figsize':(15,5)})


# In[57]:


df1["Marital_Status"].plot.hist(color='brown')
sns.set(rc={'figure.figsize':(15,5)})


# In[58]:


sns.distplot(df1["Product_Category_1"],color='black')
sns.set(rc={'figure.figsize':(15,5)})


# In[59]:


df1["Product_Category_2"].plot.hist(color='green')
sns.set(rc={'figure.figsize':(15,5)})


# In[60]:


df1["Product_Category_3"].plot.hist(color='orange')
sns.set(rc={'figure.figsize':(15,5)})


# ### Age vs Purchase

# In[126]:


sns.scatterplot(x='Age', y='Purchase', hue='Gender', data=df1)
sns.set(rc={'figure.figsize':(25,8)})


# In[124]:


sns.barplot(x="Age", y="Purchase", data=df1)
sns.set(rc={'figure.figsize':(25,10)})


# ### Occupation vs Purchased

# In[63]:


sns.barplot(x="Occupation", y="Purchase", data=df1)
sns.set(rc={'figure.figsize':(25,10)})


# In[125]:


sns.scatterplot(x='Occupation', y='Purchase', hue='Gender', data=df1)
sns.set(rc={'figure.figsize':(25,8)})


# In[65]:


df1.head(5)


# ### Product_category_1 vs Purchased

# In[122]:


sns.boxplot(x='Product_Category_1', y='Purchase', data=df1, width=2)
sns.set(rc={'figure.figsize':(30,30)})


# In[117]:


sns.barplot(x='Product_Category_1', y='Purchase', data=df1)
sns.set(rc={'figure.figsize':(25,10)})


# ### Product_category_2 vs Purchased

# In[68]:


sns.scatterplot(x='Product_Category_2', y='Purchase', hue='Gender', data=df1)
sns.set(rc={'figure.figsize':(25,10)})


# In[69]:


sns.barplot(x='Product_Category_2', y='Purchase', data=df1)
sns.set(rc={'figure.figsize':(25,10)})


# ### Product_category_3 vs Purchased

# In[70]:


sns.barplot(x='Product_Category_3', y='Purchase', data=df1)
sns.set(rc={'figure.figsize':(25,10)})


# ### City category pie chart

# In[116]:


category_count = df1.groupby('City_Category')['City_Category'].count()
values = category_count.values.tolist()
labels = category_count.index.tolist()
colors = ['pink', 'green', 'orange']
plt.pie(values, labels=labels, colors=colors,autopct='%2.2f%%')
plt.title('Distribution of City Categories')
plt.show()
sns.set(rc={'figure.figsize':(25,10)})


# # Q2 ANSWER 

# ### Joining both the dataset

# In[72]:


df=pd.concat([df1,df2])
df


# In[73]:


df.dropna(inplace=True)


# ### One Hot Encoding

# In[74]:


df= pd.get_dummies(df,columns=["City_Category"],drop_first=True)


# In[75]:


df.columns


# In[76]:


x= df[['User_ID', 'Gender', 'Age', 'Occupation', 'Stay_In_Current_City_Years',
       'Marital_Status', 'Product_Category_1', 'Product_Category_2',
       'Product_Category_3', 'City_Category_B', 'City_Category_C']].values


# x

# In[77]:


y = df['Purchase'].values


# In[78]:


y


# In[79]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state =25)


# In[80]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test =sc.transform(x_test)


# # Model Training

# ### Linear Regression

# In[81]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

y_pred1 = reg.predict(x_train)


# In[107]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[108]:


r2_score(y_train, y_pred1)


# ### Decision Tree Regressor

# In[109]:


from sklearn.tree import DecisionTreeRegressor
dc=DecisionTreeRegressor()


# In[110]:


dc.fit(x_train,y_train)


# In[111]:


y_pred=dc.predict(x_test)


# In[112]:


r2_score(y_test, y_pred)


# In[113]:


r2_score(y_train, y_pred1)


# ### Random Forest Regressor

# In[98]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[99]:


no_of_decision_tree = [10,20,30,40,50,60,70,80,90,100]
max_no_of_features = ['sqrt','log2']
max_depth = [6,7,8,9,10,11,12,13,14,15]
criterion_of_decision_tree = ["squared_error", "poisson"]
min_sample_split=[2,3,4,5,6]


# In[100]:


random_grid = {
    'n_estimators' : no_of_decision_tree,
    'max_features' : max_no_of_features,
    'max_depth' : max_depth,
    'criterion' : criterion_of_decision_tree,
    'min_samples_split' : min_sample_split
}


# Hyper Parameter Tuning

# In[101]:


# from sklearn.model_selection import RandomizedSearchCV
# rscv = RandomizedSearchCV(estimator = rf , param_distributions = random_grid , n_iter = 25 , cv = 5 ,n_jobs=-1)
# rscv.fit(x_train, y_train)


# In[102]:


# rscv.best_params_


# In[103]:


# rf = RandomForestRegressor(n_estimators = 100 , min_samples_split = 2, max_features =  'log2', max_depth = 14, criterion='poisson')


# In[104]:


# rf.fit(x_train,y_train)
# # y_pred = rf.predict(x_test)


# In[ ]:





# In[ ]:




