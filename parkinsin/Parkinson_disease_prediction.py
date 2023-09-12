#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv(r"C:\Users\Iddrisu Bachokun\Desktop\Python\Data\parkinsin\parkinsons.csv")
pd.set_option('display.max_columns',None)
df.head()


# In[6]:


df1 = pd.read_csv(r"C:\Users\Iddrisu Bachokun\Desktop\Python\Data\parkinsin\parkinsons.csv",usecols=["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Shimmer(dB)",
                                                                                                    "Shimmer:APQ3","Shimmer:APQ5","NHR","HNR","status"])
df1.head()


# In[5]:


df1.info()


# In[8]:


df1.describe()


# In[9]:


df1.isna().sum()


# In[12]:


df1.status.unique()


# In[15]:


df1['status'].value_counts()


# # <font color =green> Train, validation, test datasets </font>

# In[30]:


from imblearn.over_sampling import RandomOverSampler


# In[18]:


train , valid, test = np.split(df1.sample(frac=1), [int(0.6*len(df1)),int(0.8*len(df1))])


# In[19]:


def scale_dataset(dataframe):
    x = dataframe[dataframe.cols[:-1]].values
    y = dataframe[dataframe.cols[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    data = np.hstack((x,np.reshape(y,(-1,1))))
    return data, x,y 


# In[20]:


print(len(train[train['status']==1])) # gamma
print(len(train[train['status']==0])) # hadron


# # <font color =red > Oversampling </font> 

# ## <font color = blue>We see that the number of hadron is far too small compare with the gamma. This will poss a problem when we train our data in this form. We threrefore need to oversample the data to bring the smaller data to the same length as the longer data. This is very usefull whwen you don't haveenough data, and so the over sample will bump the data  up. There is also a situation that will reqiyer undersample. In this situation , the longer sample is undersampled to be of thesame length as the samaller data. This is usefull when you have so much data that undersampling will not compromise your results.</font>

# In[37]:


from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler


# In[38]:


def scale_dataset(dataframe,oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    if oversample:
        ros = RandomOverSampler()
        x , y = ros.fit_resample(x,y)
    data = np.hstack((x,np.reshape(y,(-1,1))))
    return data, x,y 


# In[39]:


def scale_dataset(dataframe,oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    if oversample:
        ros = RandomOverSampler()
        x , y = ros.fit_resample(x,y)
    data = np.hstack((x,np.reshape(y,(-1,1))))
    return data, x,y 


# In[ ]:





# In[40]:


print(len(train[train['status']==1])) # gamma
print(len(train[train['status']==0])) # hadron


# In[41]:


train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid ,y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)


# In[42]:


sum(y_train==1)


# In[43]:


sum(y_train==0)


# In[48]:


for label in df1.columns[:-1] :
    plt.hist(df1[df1["status"]==1][label], color ='blue', label = 'Disease', alpha=0.7, density= True)
    plt.hist(df1[df1["status"]==0][label], color ='red',label = 'No Disease', alpha=0.7, density= True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


# # <font color = red> KNN </font>

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# In[50]:


knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train)


# In[51]:


y_pred = knn_model.predict(x_test)


# In[52]:


print(classification_report(y_test,y_pred))


# In[53]:


input_data = (119.992,157.302,74.997,0.426,0.02182,0.03130,0.02211,21.033)
input_data_np = np.asarray(input_data)
input_data_re = input_data_np.reshape(1,-1)
pred = knn_model.predict(input_data_re)
print(pred)
if(pred[0]==0):
    print("The person has no disease")
    
else:
    print("The person has the disease")


# # <font color = red> Naive Bayes </font>

# In[54]:


from sklearn.naive_bayes import GaussianNB


# In[55]:


nb_model = GaussianNB()
nb_model.fit(x_test,y_test)


# In[57]:


y_pred = nb_model.predict(x_test)
print(classification_report(y_test,y_pred))


# # <font color = red> Logistic Regression </font>

# In[58]:


from sklearn.linear_model import LogisticRegression


# In[59]:


logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)


# In[ ]:


y_pred = logistic_model.predict(x_test)


# In[60]:


print(classification_report(y_test,y_pred))


# In[ ]:





# # <font color = red> SVM </font>

# In[61]:


from sklearn.svm import SVC
sv_model = SVC()


# In[62]:


sv_model.fit(x_train,y_train)


# In[63]:


y_pred = sv_model.predict(x_test)


# In[65]:


print(classification_report(y_test,y_pred))


# # <font color = red> Tree </font>

# In[66]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()


# In[67]:


dt_model.fit(x_train,y_train)


# In[68]:


y_pred = dt_model.predict(x_test)
print(classification_report(y_test,y_pred))


# # <font color = red> Neural Networks </font> 

# In[70]:


import tensorflow as tf


# In[75]:


def plot_history(history):
    fig ,(ax1, ax2)=plt.subplots(1,2,figsize = (10,4))
    ax1.plot(history.history['loss'],label='loss')
    ax1.plot(history.history['val_loss'],label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary_crossentropy')
    #ax1.legend()
    ax1.grid(True)
  
#def plot_accuracy(history):
    ax2.plot(history.history['accuracy'],label='accuracy')
    ax2.plot(history.history['val_accuracy'],label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    #ax2.legend()
    ax2.grid(True)
    plt.show()


# In[76]:


import tensorflow as tf
def train_model(X_train, y_train, num_nodes, dropout_prob,lr, batch_size, epochs):
    nnw_model =tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes,activation='relu',input_shape=(8,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes,activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1,activation='sigmoid')])

    nnw_model.compile(optimizer= tf.keras.optimizers.Adam(lr),loss='binary_crossentropy',
                      metrics=['accuracy'])
# Training the model
    history = nnw_model.fit(
        x_train, y_train, epochs=epochs, batch_size= batch_size, validation_split=0.2, verbose=0
    )
    return nnw_model, history


# In[77]:


east_val_loss = list('inf')
least_loss_model = None
epochs = 100
for num_nodes in [16,32,64]:
    for dropout_prob in[0,0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32,64,128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob},lr {lr},batch size {batch_size}")
                model , history = train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history)
                val_loss= model.evaluate(x_valid, y_valid)
                #if val_loss < least_val_loss:
                least_val_loss = val_loss
                least_loss_model = model


# In[ ]:




