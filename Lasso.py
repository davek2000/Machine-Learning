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

from sklearn.preprocessing import PolynomialFeatures
xPoly = PolynomialFeatures(2).fit_transform(X)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,yTrain,yTest = train_test_split(X,y,test_size=0.2)

# from sklearn.preprocessing import PolynomialFeatures
# xPoly = PolynomialFeatures(2).fit_transform(Xtrain)

from sklearn.linear_model import Lasso
# Choose C as 5
model = Lasso(alpha=(1/(2*5)))
model.fit(Xtrain,yTrain)

ypred=model.predict(Xtest)

from sklearn.metrics import mean_squared_error
lasso_error = mean_squared_error(yTest,ypred)
print("Lasso MSE: ",lasso_error)

from sklearn.metrics import r2_score
lasso_r2 = r2_score(yTest,ypred)
print("Lasso R2 score: ",lasso_r2)

# dummy
from sklearn.dummy import DummyRegressor
dum_model = DummyRegressor(strategy="mean")
dum_model.fit(Xtrain,yTrain)
dum_pred=dum_model.predict(Xtest)

dum_error = mean_squared_error(dum_pred,yTest)
print("Dummy MSE: ",dum_error)
dum_r2=r2_score(dum_pred,yTest)
print("Dummy r2 Score: ",dum_r2)


