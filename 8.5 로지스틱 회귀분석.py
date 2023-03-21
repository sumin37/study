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
cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split
x_tn, x_te, y_tn, y_te = train_test_split(x,y,random_state=0)

#데이터 표준화
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale.fit(x_tn)
x_tn_std = std_scale.transform(x_tn)
x_te_std = std_scale.transform(x_te)


from sklearn.linear_model import LogisticRegression
clf_logi_l2 = LogisticRegression(penalty='l2')
clf_logi_l2.fit(x_tn_std,y_tn)

print(clf_logi_l2.coef_) #회귀모형의 추정 계수

pred_logistic = clf_logi_l2.predict(x_te_std)
print(pred_logistic)

pred_proba = clf_logi_l2.predict_proba(x_te_std)
print(pred_proba)

from sklearn.metrics import precision_score
precision = precision_score(y_te, pred_logistic)
print(precision)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, pred_logistic)
print(conf_matrix)

from sklearn.metrics import classification_report
class_report = classification_report(y_te, pred_logistic)
print(class_report)


