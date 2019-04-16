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

# 使用するライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# csvをデータフレーム形式で読み込み
ratings = pd.read_csv('csv/rating.csv')
anime = pd.read_csv('csv/anime.csv')

# ratingのデータフレームの最初の5行を表示
ratings.head()

# animeのデータフレームの最初の5行を表示
anime.head()

# animeデータフレームをmembersの数で降順ソートして10件表示
anime.sort_values('members', ascending=False)[:10]

# animeの基本統計量の確認
round(anime.describe(), 2)

# ratindsの基本統計量確認
# ratingの-1は「アニメを見たことがあるが、ratingを付与しなかった」
round(ratings.describe(), 2)

# ratingsのヒストグラムを作成
ratings['rating'].hist(bins=11, figsize=(10, 10), color='red')

# membersの値が10000より大きいデータのみに変更
anime = anime[anime.members > 10000]
round(anime.describe(), 2)

# 欠損データの確認
anime.isnull().sum()

# 欠損データをdropna()でデータセットから取り除く
anime = anime.dropna()

# ratingの値が0以上のみを残す
ratings = ratings[ratings.rating >= 0]
round(ratings.describe(), 2)

# animeとratingsの2つのデータフレームをマージさせる
mergeddf = ratings.merge(anime, on='anime_id', suffixes=['_user', '_average'])
mergeddf.head()

# mergeddfの基本統計量確認
round(mergeddf.describe(), 2)

# 不必要な項目と重複項目を削除
mergeddf = mergeddf[['user_id', 'name', 'rating_user']]
mergeddf = mergeddf.drop_duplicates(['user_id', 'name'])
mergeddf.info()
mergeddf.head(10)

# データフレームのピボット
anime_pivot = mergeddf.pivot(index='name', columns='user_id', values='rating_user').fillna(0)
anime_pivot_sparse = csr_matrix(anime_pivot.values)

# anime_pivotの最初の10行を表示
anime_pivot.head(20)

# ここまででデータの前処理終了。ここからk近傍法を使用してレコメンド機能を作成していく

# Sklearnのライブラリを利用
knn = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')

# 前処理したデータセットでモデルを訓練
model_knn = knn.fit(anime_pivot_sparse)

# データセットのタイトルをキーワードで検索
def searchanime(string):
    print(anime_pivot[anime_pivot.index.str.contains(string)].index[0:])

searchanime('Death')

# 類似作を表示したいアニメのタイトルを設定
Anime = 'Death Note'

# 設定したアニメに対してのオススメアニメ10選表示
distance, indice = model_knn.kneighbors(anime_pivot.iloc[anime_pivot.index==Anime].values.reshape(1,-1), n_neighbors=11)
for i in range(0, len(distance.flatten())):
    if i == 0:
        print('Recommendations if you like the anime {0}:\n'.format(anime_pivot[anime_pivot.index==Anime].index[0]))
    else :
        print('{0}: {1} with distanse: {2}'.format(i, anime_pivot.index[indice.flatten()[i]], distance.flatten()[i]))













