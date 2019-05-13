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

import pandas as pd
import numpy as np

train = pd.read_csv("csv/train.csv")
test = pd.read_csv("csv/test.csv")

# trainデータフレームの確認
train.head()

# testデータフレームの確認
test.head()

# trainの次元の確認
train.shape

# testの次元の確認
test.shape

# trainの基本統計量の確認
train.describe()

# testの基本統計量の確認
test.describe()

# データセットの欠損の確認の関数定義
def describe_missing_values(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val / len(df)
    missing_table = pd.concat([null_val, percent], axis=1)
    missing_table_ren_cols = missing_table.rename(
            columns = {0 : '欠損数', 1 : '%'})
    return missing_table_ren_cols

# trainデータフレームの欠損数確認
describe_missing_values(train)

# testデータフレームの欠損数確認
describe_missing_values(test)

# Ageの欠損値を中央値で、Embarkedの欠損値を一番多い「S」でそれそれ補完
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna('S')

# 欠損データの補完ができたか確認
describe_missing_values(train)

# カテゴリカルデータの文字列を数字に変換
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

# カテゴリカルデータの文字列を数字に変換したことの確認
train.head(10)

# testデータについても同様の欠損値の補完とデータ変換
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()

# 変換後のtestデータの確認
test.head(10)

# データの事前処理終了

# scikit-learnのインポート
from sklearn import tree

# 「train」の目的変数(予測したい情報)と説明変数(予測に使う情報)の値を取得
target = train['Survived'].values
features_one = train[['Pclass', 'Sex', 'Age', 'Fare']].values

# 目的変数の確認
target

# 説明変数の確認
features_one

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数の値を取得
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

# 予測データのサイズを確認
my_prediction.shape

# 予測データの中身を確認
print(my_prediction)

# ここまでで予測完了。ここからcsvに書き出し処理。

# PassengerIdを取得
PassengerId = np.array(test['PassengerId']).astype(int)

# my_prediction(予測データ)とPassengerIdをデータフレームに落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ['Survived'])

# my_tree_one.csvとして書き出し
my_solution.to_csv('csv/my_tree_one.csv', index_label = ['PassengerId'])

# ここでver1の実装完了

# 追加項目も含めて予測モデルで使用する値を取得
features_two = train[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']].values

# 決定木の作成とアーギュメントの設定
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(
        max_depth = max_depth, 
        min_samples_split = min_samples_split, 
        random_state = 1
)
my_tree_two = my_tree_two.fit(features_two, target)

# testからver2で使う項目を取り出す
test_features_2 = test[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']].values

# ver2の決定木を使って予測結果csvに書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ['Survived'])
my_solution_tree_two.to_csv('csv/my_tree_two.csv', index_label = ['PassengerId'])





