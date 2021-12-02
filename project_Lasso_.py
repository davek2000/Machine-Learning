import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor

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
X11=df.iloc[:,13]   # Transfer fees
X11=X11/10000000    # reduce the size of the fee, otherwise the model will be unstable
X=np.column_stack((X1,X2,X3,X5,X6,X9,X10,X11))

y=df.iloc[:,12]     # Market Value
y=y/10000000        # reduce the size of the MV, otherwise the model will be unstable

#Split test and train data with ratio 2/8, adjust polynomial feature to q=2
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)
Xpoly = PolynomialFeatures(2).fit_transform(Xtrain)


#C=5 Lasso model, coefficients
model1 = Lasso(alpha=1/(2*5),max_iter=100000)
model1.fit(Xpoly, ytrain)
#print(model1.intercept_, model1.coef_)


#final test of the unseen data set
X_test_poly = PolynomialFeatures(2).fit_transform(Xtest)
ypred1 = model1.predict(X_test_poly)
error_final=mean_squared_error(ytest,ypred1)    #this is using mean square error as evaluation
error_final2=r2_score(ytest,ypred1)              #this is using R2 score as evaluation
print(error_final)
print(error_final2)

#basline predictor
dummy = DummyRegressor(strategy="mean").fit(Xpoly, ytrain)
ydummy = dummy.predict(X_test_poly)
error_base=mean_squared_error(ytest,ydummy)
error_base2=r2_score(ytest,ydummy)
print(error_base)
print(error_base2)

