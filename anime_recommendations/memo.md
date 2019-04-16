# anime_recommendations Memo

## データの内容

### anime.csv

| anime_id | name | genre | type | episodes | rating | members |
|----------|------|-------|------|----------|--------|---------|
| 各アニメのユニークid | アニメタイトル | アニメのカテゴリ | メディアタイプ | アニメのエピソード数 | 平均レーティング | 当該アニメのグループに参加するユーザー数 |

### ratings.csv

| user_id | anime_id | rating |
|---------|----------|--------|
| ユーザーのユニークid | 当該ユーザーがレートしたアニメid | 当該ユーザーのレーティング |

<br>
<br>

## 処理の流れ

### 下準備

- ライブラリのインポート  
- CSVの読み込み  
- データフレームの確認  

<br>

### データの前処置

- リコメンドするアニメを人気アニメ(10000人以上がグループに属している)に限定  
- 欠損データの取り除き  
  - dropna() メソッドを使用  
- 評価がついているデータのみにデータトリミング  
  - 評価が -1 (見たけど評価しなかった) のデータをratingsデータフレームから削除  
- データフレームのマージ  
  - anime_idを軸にして両データフレームをマージ  
- 不必要なデータの削除  
  - 今回は、user_id, name, rating_user のみ使用。  
  - drop_duplicates メソッドで同じ人が2度以上同じ映画を評価しているデータを削除する。  
- マージしたデータフレームからピボットの作成  
  - インデックスを name、カラムを user_id、値を rating_user としたピボットを作成。  
  - 欠損値(NaN)の部分を 0 で置換。(fillna メソッドを使用)  

<br>

### モデルの構築

- Sklearnを使用してk近傍法でモデルを訓練  

<br>

### モデルの評価

- 訓練モデルを使用し、テストデータを検証  


<br>
<br>


## 前処理 -Memo-

### DataFrameのマージについて

```python
merge(left, right, how='inner', on=None, left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('_x', '_y'), copy=True, indicator=False)
```

| 引数        | 定義 |
|-------------|------|
| left        | データフレームオブジェクト。 |
| right       | マージするデータフレームオブジェクト。 |
| on          | 結合に用いる行の名前。left と right のデータフレーム両方に存在する必要あり。 |
| left_on     | left のデータフレームでキーとして用いる列名、または配列。 |
| right_on    | right のデータフレームでキーとして用いる列名、または配列を選択。 |
| left_index  | True を設定すると、left のデータフレームの行ラベルを結合のキーとして用いる。MultiIndex (階層的なインデックス構造) を持つデータフレームの場合、階層数を left と right で合わせる必要がある。 |
| right_index | left_index と同じ |
| how         | ‘left’, ‘right’, ‘outer’, ‘inner’ のいずれかを設定。 (デフォルトは “inner”) |
| sort        | True を設定すると、結合後のデータフレームをソート。(デフォルトは True) |
| suffixes    | 同一のカラム名が存在した場合に、後ろに文字列を追加して区別する。 (デフォルトは ‘_x’, ‘_y’) |
| copy        | 常に与えられたデータフレームをコピーする。場合によっては、False に設定すると、パフォーマンスやメモリの使用量を向上できる場合がある。 (デフォルトは True) |
| indicator   | \_merge という名前のカラムを出力後のデータフレームに追加し、結合前の行に関する情報を格納。 |

<br>
<br>

### 重複した行の削除

```python
df.drop_duplicates(subset='state', keep='last', inplace=True)
```

**drop_duplicates()**  
- keep  
  - 残す行を選択。デフォルトは `keep='first'` 。つまり重複の最初の行が残されて次から重複として削除される。  
  - `keep='last'` とすると最後の行が残される。  
- subset  
  - 重複を判定する列を指定  
  - デフォルトでは全ての列要素が一致しているときのみ重複とみなされる。  
  - リストで複数の列を取ることも可能  
- inplace  
  - デフォルトでは重複した行が削除された新たなデータフレームが返される。  
  - `inplace=True` にすると元のデータフレームから重複した行が削除される。  

<br>
<br>

### 疎行列クラス

```python
anime_pivot_sparse = csr_matrix(anime_pivot.values)
```


今回のようにデータの90%以上が0で構成されている行列のことを疎行列という。  
疎行列はデータ量が大きく、そのままではCPUキャッシュやメモリに乗り切らずに非効率な計算になってしまうことが多い。  

こうした疎行列を効率的に扱うために、Pythonではscipy.spareがよく使われる。  

その中の、**csr_matrix** は行ごとに圧縮されており、行単位でなら高速にアクセスできる。  

↓ 参考URL  
[Python: scipy.sparseで疎行列計算する](https://ohke.hateblo.jp/entry/2018/01/07/230000)  

<br>
<br>

## モデル構築 -Memo-

### k近傍法

- 教師あり学習の1種  
- 新しい点を既存のデータとの距離に基づき分類する  
- あるオブジェクトの分類をその近傍オブジェクト群のk個の投票によって決定する  

↓ NearestNeighbors・KNeighborsClassifier・KNeighborsRegressor の違い  
[sklearn.neighborsの使い方調査](http://gratk.hatenablog.jp/entry/2017/12/10/205033)  

簡単に分けると、  
- NearestNeighbors  
  - 教師なし  
- KNeighborsClassifier  
  - 教師あり: 分類  
- KNeighborsRegressor  
  - 教師あり: 回帰  

















