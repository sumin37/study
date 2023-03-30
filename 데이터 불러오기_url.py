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
import pandas as pd

file_url="https://media.githubusercontent.com/media/musthave-ML10/data_source/main/salary.csv"

sample = pd.read_csv(file_url)
# -

sample.head()

sample.info() #요약 정보

#통계정보
sample.describe()

sample_df = pd.read_csv(file_url, index_col=0)

sample_df

# +
file = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/sample_df.csv"

sample_df = pd.read_csv(file, index_col=0)
# -

sample_df

#칼럼기준 index
sample_df['var_1']

sample_df[['var_1','var_2']]

# 행기준 indexing
sample_df.loc['a']

sample_df.loc[['a','b','c']]

sample_df.iloc[0:3,2:4]

#칼럼 제거
sample_df.drop('var_1',axis=1)

sample_df.drop(['var_1','var_2'],axis=1)

#인덱스 변경
sample_df.reset_index()

sample_df.reset_index(drop=True)

sample_df.set_index('var_1')

# 변수별 계산
sample_df.sum()

sample_df.aggregate(['sum','mean'])

#group 
file1 = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/iris.csv"
iris = pd.read_csv(file1)

iris.groupby('class').mean()

iris.groupby('class').agg(['count','mean'])

iris['class'].unique()

iris['class'].nunique()

iris['class'].value_counts()

# # 데이터 프레임 합치기

left_url = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/left.csv"
right_url = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/right.csv"

left = pd.read_csv(left_url)
right = pd.read_csv(right_url)

left

left.merge(right)

left.merge(right,how='outer')

left.merge(right,how='left')

left.join(right)

left.drop('key', axis=1).join(right.drop('key',axis=1))

left = left.set_index('key')
right=right.set_index('key')

left.join(right)


