import numpy as np
import pandas as pd

#df=pd.read_csv("Defenders.csv")
df=pd.read_csv("Midfielders.csv")

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

poly_range = [1,2,3,4]

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

        dum_temp.append(mean_squared_error(y[test],dum_pred))

    print("Poly i: ",poly_i)
    
    print("Logistic Regression MSE: ",np.array(lr_temp).mean())
    print("Logistic Regression MSE std: ",np.array(lr_temp).std())

    print("Dummy model MSE: ",np.array(dum_temp).mean())
    print("Dummy model MSE std: ",np.array(dum_temp).std())

    print()

# Defenders.csv
#-------------------------
# Best poly value is 1

# MSE of 179.23
# std of 250.16

# dummy model has MSE of 247.55
# and std MSE of 344.40

# Midfielders.csv
#--------------------------
# Best poly value is 1

# MSE of 313.97
# std of 435.69

# dummy model has MSE of 372.64
# and std MSE of 530.02

# Forwards.csv
#-------------------------
