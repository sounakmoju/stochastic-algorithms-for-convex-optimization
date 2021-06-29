import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from random import seed
from random import randrange
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X=load_boston().data
Y=load_boston().target
scaler=preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=pd.DataFrame(data=X_train,
columns=load_boston().feature_names)
X_train['price']=list(y_train)
X_test=pd.DataFrame(data=X_test,columns=load_boston().feature_names)
X_test['price']=list(y_test)
#print(X_train.head())
def svrg_regressor(X,y,learning_rate=0.2, n_epochs=2, k=40,l=20):
    w=np.random.randn(1,13)
    b=np.random.randn(1,1)

    epoch=0
    loss=0
    y_pred=[]
    sq_loss=[]
    w_=0
    b_=0
    batch=X.sample(k)
    while epoch<n_epochs:
        X_tr=batch.iloc[:,0:13].values
        y_tr=batch.iloc[:,-1].values
        mu_w=0
        mu_b=0
        w_=w
        b_=b
        
        for i in range(k):
            #print(y_tr[i].shape)
            mu_w+=(-2* X_tr[i]) * (y_tr[i] - np.dot(X_tr[i],w_.T) - b_)
            mu_b+=(-2)*(y_tr[i] - np.dot(X_tr[i],w_.T) - b_)
        mu_w=mu_w/k
        mu_b=mu_b/k
        w1=w_
        b1=b_
        
        for _ in range(l):
            
            index =randrange(k)#for non repetative make a list and delete the element and sample again
            phi_w1=(-2* X_tr[index]) * (y_tr[index] - np.dot(X_tr[index],w_.T) - b_)
            phi_b1=(-2)*(y_tr[index] - np.dot(X_tr[index],w_.T) - b_)
            phi_w11=(-2* X_tr[index]) * (y_tr[index] - np.dot(X_tr[index],w1.T) - b1)
            phi_b11=(-2)*(y_tr[index] - np.dot(X_tr[index],w1.T) - b1)
            w1=w1-learning_rate*(phi_w11-phi_w1+mu_w)
            b1=b1-learning_rate*(phi_b11-phi_b1+mu_b)
            
        epoch+=1
        #learning_rate = learning_rate/1.02
        w_=w1
        b_=b1
    return w_,b_
w,b = (svrg_regressor(X_train,y_train))
print(w)
        
    
