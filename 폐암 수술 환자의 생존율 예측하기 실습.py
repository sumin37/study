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

#딥러닝을 구동하는 데 필요한 케라스 함수 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import tensorflow as tf

#실행때마다 같은 결과 나오게 설정
np.random.seed(3)
tf.random.set_seed(3)

#data read
Data_set=np.loadtxt("/Users/sumin/Downloads/ThoraricSurgery.csv",delimiter=",")

Data_set

#환자의 기록과 수술 결과를 X와 Y로 구분하여 저장
X = Data_set[:,0:17]
Y = Data_set[:,17]

#딥러닝 구조를 결정(모델 설정하고 실행)
model = Sequential()
model.add(Dense(30,input_dim=17, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y,epochs=100, batch_size=10)

# +
#결과의 의미 : loss = 예측이 실패할 확률, accuracy = 예측이 성공할 확률.
