import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyRegressor

csv_name="Midfielder.csv"

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


C_range = [0.0001,0.001,0.01,0.1,0.5,1,5,10,50,100,500]

poly_range = [1,2]
mean_error=[]
std_error=[]

# for Ci in C_range:
for poly_i in poly_range:
    # model = Lasso(alpha=(1/(2*Ci)),max_iter=100000)      # find optimum C val
    
    # model = Lasso(alpha=(1/(2*10)),max_iter=100000)    # Defenders.csv
    model = Lasso(alpha=(1/(2*1)),max_iter=100000)    # Forwards.csv & Midfielder.csv

    xPoly = PolynomialFeatures(poly_i).fit_transform(X) # find optimum poly val

    # xPoly = PolynomialFeatures(1).fit_transform(X)  # Defenders.csv AND Midfielders.csv

    dum_model = DummyRegressor(strategy="mean")

    lasso_temp=[]; dum_temp=[]
    for train,test in kf.split(xPoly):
        model.fit(xPoly[train],y[train])
    
        ypred = model.predict(xPoly[test])

        from sklearn.metrics import mean_squared_error
        lasso_temp.append(mean_squared_error(y[test],ypred))

        dum_model.fit(xPoly[train],y[train])

        dum_pred = dum_model.predict(y[test])

        dum_temp.append(mean_squared_error(y[test],dum_pred))
    
    
    mean_error_num = np.array(lasso_temp).mean()
    std_error_num = np.array(lasso_temp).std()

    mean_error.append(mean_error_num)
    std_error.append(std_error_num)

    
    # print("Ci = ",Ci)
    print("poly_i = ",poly_i)
    
    print("Lasso MSE: ",np.array(lasso_temp).mean())
    print("Lasso MSE standard deviation",np.array(lasso_temp).std())

    print("Dummy MSE: ",np.array(dum_temp).mean())
    print("Dummy MSE standard deviation",np.array(dum_temp).std())

fig1=plt.figure()
ax = fig1.add_subplot(1, 1, 1)
plt.errorbar(poly_range,mean_error,yerr=std_error)
plt.xlabel('q parameter'); plt.ylabel('Mean squared error')
# ax.set_xscale('log')
# plt.title("%s : Find the optimum C parameter" % csv_name)
plt.title("%s : Find the optimum degree of polynomial" % csv_name)
plt.show()

fig2=plt.figure()  
plt.scatter(transfer_fees,y,marker='+',color='red')
  
# Model: Transfer fee vs Predicted Market Value
model2 = Lasso(alpha=(1/(2*1)),max_iter=100000)  
xPoly2 = PolynomialFeatures(1).fit_transform(X) 
model2.fit(xPoly2,y)
ypred2 = model2.predict(xPoly2)
plt.scatter(transfer_fees,ypred2,facecolors='none',edgecolors='b')
plt.xlabel("Transfer Fees"); plt.ylabel("Market Value")
plt.legend(["Actual Market Value","Predicted Market Value"])
plt.title("%s : Plot of Actual Market Values vs Predicted Market Values when C=1, poly = 1"%(csv_name ))
plt.show()

# Defenders.csv
#-------------------------
# Best C value:10
# Best poly value:1

# MSE:
# std:

# dummy model has MSE of 247.55
# and std MSE of 344.40

# Midfielders.csv
#--------------------------
# Best C value:
# Best poly value:

# MSE:
# std:

# dummy model has MSE of 372.64
# and std of 530.02

# Forwards.csv
#-------------------------
