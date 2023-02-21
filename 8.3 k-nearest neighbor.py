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
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_tn, x_te, y_tn, y_te = train_test_split(x,y,random_state=0)

from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale.fit(x_tn)
x_tn_std = std_scale.transform(x_tn)
x_te_std = std_scale.transform(x_te)

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(x_tn_std, y_tn)

# 데이터 예측
knn_pred = clf_knn.predict(x_te_std)
print(knn_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te,knn_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te,knn_pred)
print(conf_matrix)

from sklearn.metrics import classification_report
class_report = classification_report(y_te,knn_pred)
print(class_report)


