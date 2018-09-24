# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:33:07 2018

@author: wdz
"""

#Raschaka Chap 10 Predicting Continous Target Variables w/ Regression Analysis
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor




df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')





# ## Visualizing the important characteristics of a dataset
"""
Part 1: Exploratory Data Analysis
Describe the data sufficiently using the methods and visualizations that we used previously in Module 3 and again this week.  
Include any output, graphs, tables, heatmaps, box plots, etc.  Label your figures and axes. DO NOT INCLUDE CODE!
"""




print(df.head())
print(df.tail())

summary = df.describe()
print(summary)


#For your EDA you should produce a table of summary statistics for each of the 13 explanatory variables. 
# I would then produce a 13x13 correlation matrix, which could be displayed as a heatmap.  You can try adjusting the font smaller, or the size larger or set ANNOT=False if it doesn't look good.
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols =  ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT']

corr_matrix = df.iloc[:, :-1].corr()
print(corr_matrix)
_ =sns.heatmap(corr_matrix,cbar=True,annot=False,square=True,fmt='.2f',
                annot_kws={'size':10},yticklabels=cols,
                xticklabels=cols)


"""Alternatively below
cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
_ =sns.heatmap(cm,cbar=True,annot=False,square=True,fmt='.2f',
                annot_kws={'size':10},yticklabels=cols,
                xticklabels=cols)
"""
plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()


#boxplot
df_scaled = preprocessing.scale(df)
plt.figure()
plt.boxplot(df_scaled, labels = df.columns)
plt.xlabel('features')
plt.ylabel('normalized range')
plt.show()

#scatter plot
#sns.pairplot(df[df.columns], size = 2.5)
#plt.tight_layout()
#
#plt.show()
# # Implementing an ordinary least squares linear regression model

# ...

# ## Solving regression for regression parameters with gradient descent



class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)



def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor = 'white', s = 70)
    plt.plot(X, model.predict(X), color = 'black', lw = 2)
    return None

"""
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

plt.show()

num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
#print(price_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))

print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

#plt.savefig('images/10_07.png', dpi=300)
plt.show()

#alternative way but not cost efficient due to inv
# w = (X.T * X)^(-1)X.T*y

Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))


print('Slope: %.3f' % w[1])
print('Intercept: %.3f' % w[0])

ransac = RANSACRegressor(LinearRegression(), max_trials = 100,
                         min_samples = 50, loss = 'absolute_loss',
                         residual_threshold = 5.0, random_state = 0)
ransac.fit(X,y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

"""



X = df.iloc[:, :-1].values
y = df['MEDV'].values.reshape(-1, 1)
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_std,y_std, test_size=0.2, 
                                                    random_state=42)

slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


coefs = np.zeros(X.shape[1]+1)
coefs[:-1] = slr.coef_
coefs[-1]=  slr.intercept_
featureIntercept_ind = df.columns.tolist()[:-1]+['linreg intercept']
coef_df = pd.DataFrame(coefs, index= featureIntercept_ind, columns = ['linreg coef'])
print(coef_df)


def residual_plot(y_train, y_train_pred, y_test, y_test_pred, title = None):
    
    #x_min = min(min(y_train_pred), min(y_test_pred))-0.5
    #x_max = max(max(y_train_pred), max(y_test_pred))+0.5
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()

residual_plot(y_train, y_train_pred, y_test, y_test_pred, title = 'residual plot')

print('MSE train: %.3f, test: %.3f' %(mean_squared_error(y_train,y_train_pred)
, mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred)
,r2_score(y_test, y_test_pred)))



def model_result(regressor, X_train, X_test, y_train, y_test):
    y_train_pred = regressor.predict(X_train)
    y_train_pred=y_train_pred.reshape((-1,1))
    y_test_pred=regressor.predict(X_test)
    y_test_pred=y_test_pred.reshape((-1,1))
    print('coef:', regressor.coef_)
    print('intercept:', regressor.intercept_)
    print('MSE train: %.3f, test: %.3f' %(mean_squared_error(y_train,y_train_pred)
, mean_squared_error(y_test, y_test_pred)))

    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred)
,r2_score(y_test, y_test_pred)))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.show()
    
print('LR with lasso regulization', '\n')
score = []
alpha_space = np.logspace(-9, 0, 10)
for i, al in enumerate(alpha_space):
    try_lasso = Lasso(alpha = al)
    try_lasso.fit(X_train, y_train)
    y_test_pred = try_lasso.predict(X_test)
    score.append(r2_score(y_test,y_test_pred))
best_alpha_ind = [i for i, scor in enumerate(score) if scor == max(score)]
print(best_alpha_ind)    

print('the best alpha is :' , alpha_space[best_alpha_ind[0]])
    

lasso = Lasso(alpha = alpha_space[best_alpha_ind[0]])
lasso.fit(X_train, y_train)
model_result(lasso, X_train, X_test, y_train, y_test)


    
print('LR with ridge regulization', '\n')
score2 = []
alpha_space = np.logspace(-9, 0, 10)
for i, ri in enumerate(alpha_space):
    try_ridge = Ridge(alpha = ri)
    try_ridge.fit(X_train, y_train)
    y_test_pred = try_ridge.predict(X_test)
    score2.append(r2_score(y_test,y_test_pred))
best_alpha_ind2 = [i for i, scor in enumerate(score2) if scor == max(score2)]
print(best_alpha_ind2)    

print('the best alpha is :' , alpha_space[best_alpha_ind2[0]])
    

ridge = Ridge(alpha = alpha_space[best_alpha_ind2[0]])
ridge.fit(X_train, y_train)
model_result(ridge, X_train, X_test, y_train, y_test)



print('Linear Regression with elastic girl I mean elasticNet','\n')
#fixed our alpha = 1



"""sklearn document
The parameter l1_ratio corresponds to alpha in the glmnet R package while alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio = 1 is the lasso penalty. 
Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha.
"""
score3=[]
l1_range = np.arange(0.01,1,0.01).tolist()
for l1 in l1_range:
    try_elasticnet = ElasticNet(alpha=1, l1_ratio=l1)
    try_elasticnet.fit(X_train,y_train)
    y_test_pred=try_elasticnet.predict(X_test)
    score3.append(r2_score(y_test,y_test_pred))

best_alpha_ind3 = [i for i, scor in enumerate(score3) if scor == max(score3)]

print('The best ratio is: ', l1_range[best_alpha_ind3[0]])

elasticnet = ElasticNet(alpha = 1, l1_ratio =l1_range[best_alpha_ind3[0]] )
elasticnet.fit(X_train, y_train)
model_result(elasticnet, X_train, X_test, y_train, y_test)


print("My name is {Dizhou Wu}")
print("My NetID is: {dizhouw2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    











































































