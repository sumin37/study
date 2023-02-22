# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()

x = boston.data
y = boston.target

x_tn, x_te, y_tn,y_te = train_test_split(x,y,random_state=1)

std_scale = StandardScaler()
std_scale.fit(x_tn)
x_tn_std = std_scale.transform(x_tn)
x_te_std = std_scale.transform(x_te)

clf_lr = LinearRegression()

clf_lr.fit(x_tn_std,y_tn)

print(clf_lr.coef_)
print(clf_lr.intercept_)

clf_ridge = Ridge(alpha=1)
clf_ridge.fit(x_tn_std,y_tn)
print(clf_ridge.coef_)
print(clf_ridge.intercept_)

clf_lasso = Lasso(alpha=0.01)
clf_lasso.fit(x_tn_std,y_tn)

print(clf_lasso.coef_)
print(clf_lasso.intercept_)

clf_elastic = ElasticNet(alpha=0.01, l1_ratio=0.01)
clf_elastic.fit(x_tn_std,y_tn)
print(clf_elastic.coef_)
print(clf_elastic.intercept_)

pred_lr = clf_lr.predict(x_te_std)
pred_ridge = clf_ridge.predict(x_te_std)
pred_lasso = clf_lasso.predict(x_te_std)
pred_elastic = clf_elastic.predict(x_te_std)

print(r2_score(y_te, pred_lr))
print(r2_score(y_te,pred_ridge))
print(r2_score(y_te,pred_lasso))
print(r2_score(y_te,pred_elastic))

print(mean_squared_error(y_te, pred_lr))
print(mean_squared_error(y_te, pred_ridge))
print(mean_squared_error(y_te, pred_lasso))
print(mean_squared_error(y_te,pred_elastic))


