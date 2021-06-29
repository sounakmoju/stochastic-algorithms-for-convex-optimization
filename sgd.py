X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=pd.DataFrame(data=X_train,
columns=load_boston().feature_names)
X_train['price']=list(y_train)
X_test=pd.DataFrame(data=X_test,columns=load_boston().feature_names)
X_test['price']=list(y_test)
#print(X_train.head())
def sgd_regressor(X,y,learning_rate=0.2, n_epochs=1000, k=30):
    w=np.random.randn(1,13)
    b=np.random.randn(1,1)

    epoch=0
    loss=0
    y_pred=[]
    sq_loss=[]
    while epoch<n_epochs:
        batch=X.sample(k)
        X_tr=batch.iloc[:,0:13].values
        y_tr=batch.iloc[:,-1].values
        l_w=0
        l_b=0
        for i in range(k):
            #print(y_tr[i].shape)
            l_w+=(-2* X_tr[i]) * (y_tr[i] - np.dot(X_tr[i],w.T) - b)
            l_b+=(-2)*(y_tr[i] - np.dot(X_tr[i],w.T) - b)
        w=w-(learning_rate/k)*l_w
        l_b=l_b-(learning_rate/k)*l_b
        for i in range(k):
            y_predicted = np.dot(X_tr[i],w.T)
            y_pred.append(y_predicted)
            #loss = mean_squared_error(y_pred, y_tr[i])
            #print(loss)
        epoch+=1
        learning_rate = learning_rate/1.02
    return w,b
w,b = (sgd_regressor(X_train,y_train))
print(w)
        
