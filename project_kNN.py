import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

#Import Data
df=pd.read_csv("Defenders.csv")
# print(df.head())
X1=df.iloc[:,2]     # Age
X2=df.iloc[:,3]     # Games Played
X3=df.iloc[:,4]     # Goals
X4=df.iloc[:,5]     # Own Goals
X5=df.iloc[:,6]     # Assists
X6=df.iloc[:,7]     # Yellow Cards
X7=df.iloc[:,8]     # Red Cards
X8=df.iloc[:,9]     # Second Yellow
X9=df.iloc[:,10]    # Subbed On 
X10=df.iloc[:,11]   # Subbed Off
X11=df.iloc[:,13]   # Transfer fees
X11=X11/10000000    # reduce the size to make sure model is stable
X=np.column_stack((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11))
y=df.iloc[:,12]     # Market Value
y=y/10000000        # reduce the size to make sure model is stable

#Split test and train data with ratio 20% / 80%
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)

#train knn model with different n_neighbor, weights is using uniform
model1 = KNeighborsRegressor(n_neighbors=6,weights='uniform').fit(Xtrain, ytrain)
ypred1 = model1.predict(Xtest)

#calculate the error of the knn model
error_final1=mean_squared_error(ytest,ypred1)
print(error_final1)

#define gaussian weight, will be used in knn models
def gaussian_kernel2(distances):
    weights=np.exp(-0.5*(distances**2))
    return weights
    
def gaussian_kernel20(distances):
    weights=np.exp(-0.05*(distances**2))
    return weights

def gaussian_kernel200(distances):
    weights=np.exp(-0.005*(distances**2))
    return weights

#train knn models using different gaussian weights
model2 = KNeighborsRegressor(n_neighbors=6,weights=gaussian_kernel2).fit(Xtrain, ytrain)
ypred2 = model2.predict(Xtest)
model3 = KNeighborsRegressor(n_neighbors=6,weights=gaussian_kernel20).fit(Xtrain, ytrain)
ypred3 = model3.predict(Xtest)
model4 = KNeighborsRegressor(n_neighbors=6,weights=gaussian_kernel200).fit(Xtrain, ytrain)
ypred4 = model4.predict(Xtest)

#also calculate the predictions errors
error_final2=mean_squared_error(ytest,ypred2)
print(error_final2)
error_final3=mean_squared_error(ytest,ypred3)
print(error_final3)
error_final4=mean_squared_error(ytest,ypred4)
print(error_final4)

