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

# +
#import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

print(accuracy_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred, normalize=False))
# -

# ## confusion matrix

from sklearn.metrics import confusion_matrix
y_true = [2,0,2,2,0,1]
y_pred = [0,0,2,2,0,2]
confusion_matrix(y_true, y_pred)


from sklearn.metrics import classification_report
y_true = [0,1,2,2,0]
y_pred = [0,0,2,1,0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names = target_names))

from sklearn.metrics import mean_absolute_error
y_true = [3,-0.5,2,7]
y_pred = [2.5,0.0,2,8]
print(mean_absolute_error(y_true, y_pred))

from sklearn.metrics import mean_squared_error
y_true = [3,-0.5,2,7]
y_pred = [2.5,0.0,2,8]
print(mean_squared_error(y_true, y_pred))

# +
from sklearn.metrics import r2_score
y_true = [3,-0.5,2,7]
y_pred = [2.5,0.0,2,8]

print(r2_score(y_true, y_pred))
# -

from sklearn.metrics import silhouette_score
X = [[1,2], [4,5],[2,1],[6,7],[2,3]]

labels = [0,1,0,1,0]

sil_score = silhouette_score(X,labels)
print(sil_score)


