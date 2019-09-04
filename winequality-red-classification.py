
# coding: utf-8

# In[81]:


import pandas as pd


# In[99]:


wine = pd.read_csv("C:/wine-red.csv")


# In[100]:


wine.head()


# In[121]:


X = wine.drop("5",axis=1)


# In[122]:


y = wine["5"]


# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[125]:


from sklearn.preprocessing import StandardScaler


# In[126]:


scaler = StandardScaler()


# In[127]:


scaler.fit(X_train)


# In[128]:


X_train =scaler.transform(X_train)


# In[129]:


X_test =scaler.transform(X_test)


# In[130]:


from sklearn.neural_network import MLPClassifier


# In[131]:


mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)


# In[132]:


mlp.fit(X_train,y_train)


# In[133]:


predictions = mlp.predict(X_test)


# In[134]:


print(predictions)


# In[135]:


from sklearn.metrics import classification_report,confusion_matrix


# In[136]:


print(confusion_matrix(y_test,predictions))


# In[137]:


print(classification_report(y_test,predictions))


# In[ ]:




