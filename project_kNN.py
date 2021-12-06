import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt

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

csv_name="Defenders.csv"
# csv_name="Midfielders.csv"

df=pd.read_csv(csv_name)

age = df.iloc[:,2]
games_played=df.iloc[:,3]
goals = df.iloc[:,4]
own_goals = df.iloc[:,5]
assists = df.iloc[:,6]
yellow_cards = df.iloc[:,7]
red_cards=df.iloc[:,8]
second_yellows=df.iloc[:,9]
subbed_on = df.iloc[:,10]
subbed_off=df.iloc[:,11]

transfer_fees=df.iloc[:,12]
transfer_fees = transfer_fees/1000000   # divided by a million

X = np.column_stack((age,games_played,goals,own_goals,assists,yellow_cards,
                    red_cards,second_yellows,subbed_on,subbed_off,transfer_fees))
y = df.iloc[:,13]
y = y/1000000   # divided by a million
######################################################################################### part1 tuning the k parameter
# kf = KFold(n_splits=5,shuffle=True)
# neighbor_num = [1,2,3,4,5,6,7]
# mean_error=[]
# std_error=[]
# for neighbor in neighbor_num:

#     model = KNeighborsRegressor(n_neighbors=neighbor,weights='uniform')

#     knn_temp = []; dum_temp=[]
#     for train,test in kf.split(X):
#         model.fit(X[train],y[train])

#         ypred = model.predict(X[test])

#         knn_temp.append(mean_squared_error(y[test],ypred))

#         dum_model = DummyRegressor(strategy="mean")

#         dum_model.fit(X[train],y[train])

#         dum_pred = dum_model.predict(X[test])

#         dum_temp.append(mean_squared_error(y[test],dum_pred))

 


#     mean_error_num = np.array(knn_temp).mean()
#     std_error_num = np.array(knn_temp).std()

#     mean_error.append(mean_error_num)
#     std_error.append(std_error_num)
    
#     print("number of neighbors i: ",neighbor)
    
#     print("kNN MSE: ",mean_error_num)
#     print("kNN MSE std: ",std_error_num)

#     print("Dummy model MSE: ",np.array(dum_temp).mean())
#     print("Dummy model MSE std: ",np.array(dum_temp).std())

# fig1=plt.figure()
# ax = fig1.add_subplot(1, 1, 1)
# plt.errorbar(neighbor_num,mean_error,yerr=std_error)
# plt.xlabel('k neighbor parameter'); plt.ylabel('Mean squared error')
# plt.title("%s : Find the optimum number of nearest neighbors" % csv_name)
# plt.show()

########################################################################################### part2 tuning the weighting function
mean_error=[]
std_error=[]
kf = KFold(n_splits=5,shuffle=True)
knn_temp1 = []; dum_temp=[]
knn_temp2 = [];
knn_temp3 = [];
knn_temp4 = []; 
model1 = KNeighborsRegressor(n_neighbors=6,weights='uniform')
model2 = KNeighborsRegressor(n_neighbors=6,weights=gaussian_kernel2)
model3 = KNeighborsRegressor(n_neighbors=6,weights=gaussian_kernel20)
model4 = KNeighborsRegressor(n_neighbors=6,weights=gaussian_kernel200)


for train,test in kf.split(X):
    model1.fit(X[train],y[train])
    model2.fit(X[train],y[train])
    model3.fit(X[train],y[train])
    model4.fit(X[train],y[train])
    ypred1 = model1.predict(X[test])
    ypred2 = model2.predict(X[test])
    ypred3 = model3.predict(X[test])
    ypred4 = model4.predict(X[test])
    knn_temp1.append(mean_squared_error(y[test],ypred1))
    knn_temp2.append(mean_squared_error(y[test],ypred2))
    knn_temp3.append(mean_squared_error(y[test],ypred3))
    knn_temp4.append(mean_squared_error(y[test],ypred4))

    dum_model = DummyRegressor(strategy="mean")

    dum_model.fit(X[train],y[train])

    dum_pred = dum_model.predict(X[test])

    dum_temp.append(mean_squared_error(y[test],dum_pred))
mean_error_num1 = np.array(knn_temp1).mean()
std_error_num1 = np.array(knn_temp1).std()
mean_error_num2 = np.array(knn_temp2).mean()
std_error_num2 = np.array(knn_temp2).std()
mean_error_num3 = np.array(knn_temp3).mean()
std_error_num3 = np.array(knn_temp3).std()
mean_error_num4 = np.array(knn_temp4).mean()
std_error_num4 = np.array(knn_temp4).std()
mean_error.append(mean_error_num1)
std_error.append(std_error_num1)   
mean_error.append(mean_error_num2)
std_error.append(std_error_num2) 
mean_error.append(mean_error_num3)
std_error.append(std_error_num3) 
mean_error.append(mean_error_num4)
std_error.append(std_error_num4) 
print("Dummy model MSE: ",np.array(dum_temp).mean())
print("Dummy model MSE std: ",np.array(dum_temp).std())

weights_function = [1,2,3,4]
fig1=plt.figure()
ax = fig1.add_subplot(1, 1, 1)
plt.errorbar(weights_function,mean_error,yerr=std_error)
plt.xlabel('weighting function'); plt.ylabel('Mean squared error')
plt.title("%s : Find the optimum weighting function" % csv_name)
plt.show()

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)
model1 = KNeighborsRegressor(n_neighbors=6,weights='uniform')
model1.fit(Xtrain,ytrain)
ypred_true = model1.predict(Xtest)
fig2=plt.figure()  
plt.scatter(Xtest[:,10],ytest,marker='+',color='red')
  


plt.scatter(Xtest[:,10],ypred_true,facecolors='none',edgecolors='b')
plt.xlabel("Transfer Fees"); plt.ylabel("Market Value")
plt.legend(["Actual Market Value","Predicted Market Value"])
plt.title("%s : Plot of Actual Market Values vs Predicted Market Values when k=6, weighting function=uniform."%(csv_name ))
plt.show()

