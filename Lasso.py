import numpy as np
import pandas as pd

df = pd.read_csv("Defenders.csv")

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

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

# from sklearn.preprocessing import PolynomialFeatures
# xPoly = PolynomialFeatures(2).fit_transform(X)

C_range = [0.1,0.5,1,5,10,50,100,500]
poly_range = [1,2,3,4]

for Ci in C_range:
#for poly_i in poly_range:
    from sklearn.linear_model import Lasso
    # 10 is optimum C value
    model = Lasso(alpha=(1/2*Ci))
    #model = Lasso(alpha=(1/(2*10)))

    from sklearn.preprocessing import PolynomialFeatures
    xPoly = PolynomialFeatures(2).fit_transform(X)
    #xPoly = PolynomialFeatures(poly_i).fit_transform(X)

    from sklearn.dummy import DummyRegressor
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
    print("Ci = ",Ci)
    #print("poly_i = ",poly_i)
    print("Lasso MSE: ",np.array(lasso_temp).mean())
    print("Lasso MSE standard deviation",np.array(lasso_temp).std())

    print("Dummy MSE: ",np.array(dum_temp).mean())
    print("Dummy MSE standard deviation",np.array(dum_temp).std())
    print()
# Best C value is 10
# Best poly value is 2
