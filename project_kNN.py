import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
#csv_name="Defenders.csv"
csv_name="Midfielders.csv"

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

kf = KFold(n_splits=5,shuffle=True)
neighbor_num = [1,2,3,4,5,6,7]
mean_error=[]
std_error=[]

for neighbor in neighbor_num:

    model = KNeighborsRegressor(n_neighbors=neighbor,weights='uniform')

    knn_temp = []; dum_temp=[]
    for train,test in kf.split(X):
        model.fit(X[train],y[train])

        ypred = model.predict(X[test])

        knn_temp.append(mean_squared_error(y[test],ypred))

        dum_model = DummyRegressor(strategy="mean")

        dum_model.fit(X[train],y[train])

        dum_pred = dum_model.predict(X[test])

        dum_temp.append(mean_squared_error(y[test],dum_pred))

    # Plot predictions vs real data
    
    # Real data: Transfer fee vs Market value
    plt.scatter(transfer_fees,y,marker='+',color='red')
    
    # Model: Transfer fee vs Predicted Market Value
    
    ypred = model.predict(X)
    plt.scatter(transfer_fees,ypred,facecolors='none',edgecolors='b')

    # Dummy Model: Transfer Fee vs Predicted Market Value
    # dum_pred = dum_model.predict(X)
    # plt.scatter(transfer_fees,dum_pred,facecolors='none',edgecolors='b')


    plt.xlabel("Transfer Fees"); plt.ylabel("Market Value")
    plt.legend(["Actual Market Value","Predicted Market Value"])
    
    plt.title("%s : Plot of Actual Market Values vs Predicted Market Values when number of neighbors = %i"%(csv_name ,neighbor))
    #plt.title("Plot of Actual Market Values vs Predicted Market Values with Dummy Model")

    #plt.show()

    mean_error_num = np.array(knn_temp).mean()
    std_error_num = np.array(knn_temp).std()

    mean_error.append(mean_error_num)
    std_error.append(std_error_num)

    print("number of neighbors i: ",neighbor)
    
    print("kNN MSE: ",mean_error_num)
    print("kNN MSE std: ",std_error_num)

    print("Dummy model MSE: ",np.array(dum_temp).mean())
    print("Dummy model MSE std: ",np.array(dum_temp).std())

    print()

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

