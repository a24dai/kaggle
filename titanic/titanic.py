# -*- coding: utf-8 -*-
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
# %matplotlib inline
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

# trainデータフレームの先頭5行を確認
train_df.head()

# Categoricalデータ：  
# - Categorical  
#   - Survived, Sex, Embarked  
# - Ordinal  
#   - Pclass  
#
# Numericalデータ：  
# - Continuout  
#   - Age, Fare  
# - Discreate  
#   - SibSp, Parch  

# trainデータフレームの末尾5行を確認
train_df.tail()

# trainデータフレームとtestデータフレームの総合情報を確認
# =>
# Cabin, Age, Embarked に欠損データあり
# 7要素がint型、5要素がstring型
train_df.info()
print('-' * 40)
test_df.info()

# trainデータフレームの基本統計量を確認
# =>
# 38%くらいの生存率、2/3以上の人は1人で乗船している
train_df.describe()

# trainデータフレームのオブジェクトデータ(srt型等)に関する情報を取得  
# cont: データ数, unique: 重複を排除したデータ数, top: 最も多く含まれるデータ, freq: そのデータが含まれる個数
# =>
# 同性同名はいない
# 65%くらいが男性
train_df.describe(include=['O'])

# ---------------------
# - データの確認把握  
# - 欠損データの補完  
#  - Age  
#  - Embarked  
# - データの精査(削除)  
#  - Ticketは22%が重複しているので関係なさそう  
#  - Cabinは欠損データが多すぎて補完仕切れない  
#  - PassengerIdとNameも生存率には関係なさそう  
# - 既存のデータから新しいデータの作成  
#  - SibSpとParchから同乗者でカラムが作れそう  
#  - NameのTitle(MrとかMsとか)からカラムが作れそう  
#  - AgeをOrdinalデータにできそう  
#  - FareもAge同様  
# - データから読み取った結果をもとに分類の追加  
#  - 女性のが生存率高い  
#  - 子供のが生存率高い  
#  - 上流階級のが生存率高い  
# ---------------------

# 以下どの要素が生存率と関連性があるかを調べる

# チケットのクラスと生存率の関連性
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 性別と生存率の関連性
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 同乗している兄弟・配偶者の数と生存率の関係性
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 同乗している親・子供の数と生存率の関係性
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 以下グラフによる生存率のその他要素の関連性の可視化

# Numericalデータの比較
# 年齢と生存率の関連性
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Numericalデータの比較
# 年齢と性別と生存率の比較
g = sns.FacetGrid(train_df, col='Survived', row='Sex')
g.map(plt.hist, 'Age', bins=20)
grid.add_legend()

# 年齢、チケットのクラスと生存率の関連性
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# 出航港、性別と生存率の関連性
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# 出航港、料金、性別と生存率の関連性
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

# 学習に使わないデータをデータフレームから削除
# チケット番号と客室番号は生存率と関係が低いとして削除する
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

# 名前と乗客IDを削除する前に名前のタイトル(MsとかMrとか)を抜き出してTitleカラムとして保存
# crosstabでカテゴリカルデータの集計結果を表示
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

# 上記のタイトルについて出現率が少ないものはRareとしてまとめてそれぞれの生存率を算出
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# カテゴリカルデータを数値データに変換

# それぞれを数値に変換、欠損値は0で補完
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# 必要な情報は名前から抜き出せたので名前と乗客IDをデータフレームから削除
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# 性別のデータを数値に変換
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

# 以降数値データの欠損値の補完
# 欠損値について他のデータを参考に値を推測して補完する方法をとる

# チケットクラス、性別、年齢のヒストグラムを確認
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# チケットクラスと性別から年齢の欠損値を推測
guess_ages = np.zeros((2, 3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head(10)

# 年齢をpd.cutで分割ごとの離散値(AgeBand)に変換して生存率を比較
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# 6歳以下の生存率が特に高いので6歳以下用のカラムを作成
for dataset in combine:
    dataset.loc[dataset['Age'] <= 6, 'IsChild'] = 1
    dataset.loc[dataset['Age'] > 6, 'IsChild'] = 0
    dataset['IsChild'] = dataset['IsChild'].astype(int)

combine = [train_df, test_df]
train_df.head()

# 上記のAgeBandを0~3としてAgeカラムに上書き
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)
train_df.head(50)

# AgeBandを削除
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.describe()

# 兄弟・配偶者カラム(SibSp)と親・子(Parch)をまとめて、FamilySizeとして定義
# SibSpとParchを消すため
# 生存率との関連性を表示
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 新たなカラムとしてお一人様かどうか(IsAlone)を定義して生存率と比較
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# IsAloneの方が使えそうなのでそっちを採用
# SibSp, Parch, Family sizeを削除
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
train_df.describe()

# ファミリーサイズをバンド化
for dataset in combine:    
    dataset.loc[ dataset['FamilySize'] == 4, 'Family'] = 0
    dataset.loc[(dataset['FamilySize'] == 3) | (dataset['FamilySize'] == 2), 'Family'] = 1
    dataset.loc[(dataset['FamilySize'] == 7) | (dataset['FamilySize'] == 1), 'Family'] = 2
    dataset.loc[(dataset['FamilySize'] == 5) | (dataset['FamilySize'] == 6), 'Family'] = 3
    dataset.loc[ (dataset['FamilySize'] == 8) | (dataset['FamilySize'] == 11), 'Family'] = 4
    dataset['Family'] = dataset['Family'].astype(int)

train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()

train_df = train_df.drop(['FamilySize'], axis=1)
test_df = test_df.drop(['FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

# 年齢とチケットクラスを掛けたAge*Classを定義
# 年齢もチケットも数が少ない方が生存率が高いのでAge*Classも小さい方が生存率が高い
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# 出航港データの変換

# もっとも出現率の高いデータを取得
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

# 2レコード欠損値があるのでそれについてはもっとも出現率の多いデータで補完
# 出航港と生存率の関連性
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 出航港データを数値データに変換
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

# 料金についての変換

# testデータフレームに欠陥データが1レコードのみあるので中央値で補完
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

# 年齢と同様に4つの要素で離散データ化(FareBand作成)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# FareをFareBandで上書き
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

train_df.describe()

# testデータフレームも同様に確認
test_df.head(10)

# --------------------
#
# ここまででデータの前処置が終了
#
# ------------------

# ## ここからModelの学習  
# - Logistic Regression  
# - KNN or k-Nearest Neighbors  
# - Support Vector Machines  
# - Naive Bayes classifier  
# - Decision Tree  
# - Random Forrest  
# - Perceptron  
# - Artificial neural network  
# - RVM or Relevance Vector Machine  
#
# 上記の全てで学習して比較
#
# ------------------------

# 学習用データの作成
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# GridSearchをインポート
from sklearn.model_selection import GridSearchCV

# GridSearchを使用して交差検証
parameters = {
    'n_estimators': [i for i in range(10, 100, 10)],
    'criterion': ['gini','entropy'],
    'max_depth': [i for i in range(1, 10, 1)],
    'min_samples_split': [2, 4, 10, 12, 16],
    'bootstrap': [True, False],
    'random_state': 1,
}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1, verbose=1)
clf.fit(X_train, Y_train)
predictor = clf.best_estimator_
best_score = clf.best_score_
best_params = clf.best_params_
print('best score: ', best_score, 'best params: ', best_params)

# 交差検証のでテストデータを予測
grid_pred = predictor.predict(X_test)

# +
# Random Forest
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, Y_train)
# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# acc_random_forest
# -

# 提出用CSVデータ作成
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": grid_pred
    })
submission.to_csv('csv/submission11.csv', index=False)









