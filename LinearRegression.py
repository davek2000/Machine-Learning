import numpy as np
import pandas as pd

#csv_name="Defenders.csv"
#csv_name="Midfielders.csv"
csv_name="Forwards.csv"

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

# Plot input features vs output value
import matplotlib.pyplot as plt
plt.scatter(games_played,y)
plt.xlabel("Games Played"); plt.ylabel("Market Value")
plt.title("%s : Games Played vs Market Value"%csv_name)
plt.show()

from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True)

poly_range = [1,2,3]
mean_error=[]
std_error=[]

for poly_i in poly_range:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    from sklearn.preprocessing import PolynomialFeatures
    xPoly = PolynomialFeatures(poly_i).fit_transform(X)

    lr_temp = []; dum_temp=[]
    for train,test in kf.split(xPoly):
        model.fit(xPoly[train],y[train])

        ypred = model.predict(xPoly[test])

        from sklearn.metrics import mean_squared_error
        lr_temp.append(mean_squared_error(y[test],ypred))

        from sklearn.dummy import DummyRegressor
        dum_model = DummyRegressor(strategy="mean")

        dum_model.fit(xPoly[train],y[train])

        dum_pred = dum_model.predict(xPoly[test])

        test_var = xPoly[test]

        dum_temp.append(mean_squared_error(y[test],dum_pred))

    # Plot predictions vs real data
    import matplotlib.pyplot as plt
    
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
    
    plt.title("%s : Plot of Actual Market Values vs Predicted Market Values when poly = %i"%(csv_name ,poly_i))
    #plt.title("Plot of Actual Market Values vs Predicted Market Values with Dummy Model")

    plt.show()

    mean_error_num = np.array(lr_temp).mean()
    std_error_num = np.array(lr_temp).std()

    mean_error.append(mean_error_num)
    std_error.append(std_error_num)

    print("Poly i: ",poly_i)
    
    print("Logistic Regression MSE: ",mean_error_num)
    print("Logistic Regression MSE std: ",std_error_num)

    print("Dummy model MSE: ",np.array(dum_temp).mean())
    print("Dummy model MSE std: ",np.array(dum_temp).std())

    print()

# # Plot how to choose hyper-parameter: degree of polynomial
# import matplotlib.pyplot as plt
# print(mean_error)


# plt.rcParams['figure.constrained_layout.use']=True
# plt.rc('font',size=18)
# plt.errorbar(poly_range, mean_error, yerr=std_error,linewidth=3)
# plt.xlabel('Degree of Polynomial'); plt.ylabel('Mean Square Error')
# plt.title("%s : Optimum degree of polynomial" % csv_name)
# plt.legend(["Linear Regression"])
# plt.show()


#-------------------------
# Defenders.csv:
# Best poly value: 1

# MSE: 106.28
# std: 21.22

# dummy model has MSE of 174.53
# and std MSE of 26.34

#--------------------------
# Midfielders.csv:
# Best poly value: 1

# MSE: 188.23
# std: 35.91

# dummy model has MSE of 262.66
# and std MSE of 27.95

#-------------------------
# Forwards.csv:
# Best poly value: 1

# MSE: 242.78
# std: 90.19

# dummy model has MSE of 370.28
# and std MSE of 135.58
