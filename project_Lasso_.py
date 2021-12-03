import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt

#csv_name="Defenders.csv"
csv_name="Midfielders.csv"

df = pd.read_csv(csv_name)

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
#C_range=[0.1]
poly_range=[1,2,3]

mean_error=[]; std_error=[]

for Ci in C_range:
#for poly_i in poly_range:
   
    model = Lasso(alpha=(1/(2*Ci)))     # find optimum C val

    #model = Lasso(alpha=(1/(2*5)))  # Defenders.csv AND Midfielders.csv

    
    #xPoly = PolynomialFeatures(poly_i).fit_transform(X) # find optimum poly_i val
    
    xPoly = PolynomialFeatures(1).fit_transform(X)      # Defenders.csv AND Midfielders.csv

    
    dum_model = DummyRegressor(strategy="mean")

    lasso_temp=[]; dum_temp=[]
    for train,test in kf.split(xPoly):
        model.fit(xPoly[train],y[train])
    
        ypred = model.predict(xPoly[test])

        lasso_temp.append(mean_squared_error(y[test],ypred))

        dum_model.fit(xPoly[train],y[train])

        dum_pred = dum_model.predict(y[test])

        dum_temp.append(mean_squared_error(y[test],dum_pred))

    # Plot predictions vs real data
    
    # Real data: Transfer fee vs Market value
    plt.scatter(transfer_fees,y,marker='+',color='red')
    
    # Model: Transfer fee vs Predicted Market Value
    
    ypred = model.predict(xPoly)
    plt.scatter(transfer_fees,ypred,facecolors='none',edgecolors='b')

    # Dummy Model: Transfer Fee vs Predicted Market Value
    # dum_pred = dum_model.predict(xPoly)
    # plt.scatter(transfer_fees,dum_pred,facecolors='none',edgecolors='b')


    plt.xlabel("Transfer Fees"); plt.ylabel("Market Value")
    plt.legend(["Actual Market Value","Predicted Market Value"])
    
    plt.title("%s : Plot of Actual Market Values vs Predicted Market Values"%(csv_name))
    #plt.title("%s : Plot of Actual Market Values vs Predicted Market Values with Dummy Model"%csv_name)

    plt.show()

    mean_error_num = np.array(lasso_temp).mean()
    std_error_num = np.array(lasso_temp).std()

    mean_error.append(mean_error_num)
    std_error.append(std_error_num)
    
    print("Ci = ",Ci)
    #print("poly_i = ",poly_i)
    
    print("Lasso MSE: ",mean_error_num)
    print("Lasso MSE standard deviation",std_error_num)

    print("Dummy MSE: ",np.array(dum_temp).mean())
    print("Dummy MSE standard deviation",np.array(dum_temp).std())



# Plot how to choose hyper-parameter: C
# import matplotlib.pyplot as plt
# print(mean_error)
# print(std_error)


# plt.rcParams['figure.constrained_layout.use']=True
# plt.rc('font',size=18)
# plt.errorbar(C_range, mean_error, yerr=std_error,linewidth=3)
# plt.xlabel('Ci'); plt.ylabel('Mean Square Error')
# plt.title("%s : Optimum C value" % csv_name)
# plt.legend(["Lasso Regression"])
# plt.show()

# Defenders.csv
#-------------------------
# Best C value: 0.1
# Best poly value: 1

# MSE:
# std:

# dummy model has MSE of 247.55
# and std MSE of 344.40

# Midfielders.csv
#--------------------------
# Best C value: 0.1
# Best poly value: 1

# MSE: 188.85
# std: 43.62

# dummy model has MSE of 262.91
# and std of 55.85

# Forwards.csv
#-------------------------

