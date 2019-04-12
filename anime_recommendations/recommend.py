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

