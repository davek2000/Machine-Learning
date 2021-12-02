

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches


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
X8=df.iloc[:,9]     # Second Yellows
X9=df.iloc[:,10]    # Subbed On
X10=df.iloc[:,11]   # Subbed Off
X=np.column_stack((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10))

y=df.iloc[:,12]     # Market Value


#Split test and train data with ratio 2/8, adjust polynomial feature to q=2
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)
Xpoly = PolynomialFeatures(1).fit_transform(Xtrain)


#C=5 Lasso model, coefficients
model1 = Lasso(alpha=1/(2*5),max_iter=100000)
model1.fit(Xpoly, ytrain)
#print(model1.intercept_, model1.coef_)


#final test of the unseen data set
X_test_poly = PolynomialFeatures(1).fit_transform(Xtest)
ypred1 = model1.predict(X_test_poly)
error_final=mean_squared_error(ytest,ypred1)
print(error_final)

#basline predictor
constant=np.mean(y);
length=len(y)
baseline_pred=np.empty(length)
baseline_pred.fill(constant)
error_base=mean_squared_error(baseline_pred,y)
print(error_base)

