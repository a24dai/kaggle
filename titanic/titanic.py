# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# 各ライブラリのインポート
# データ分析ライブラリ
import pandas as pd
import numpy as np
import random as rnd
# データ可視化ライブラリ
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# 機械学習ライブラリ
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# csvをデータフレームとして読み込み
train_df = pd.read_csv('csv/train.csv')
test_df = pd.read_csv('csv/test.csv')
combine = [train_df, test_df]

# trainデータフレームのカラム確認
print(train_df.columns.values)

# trainデータフレームの確認
train_df.head()

