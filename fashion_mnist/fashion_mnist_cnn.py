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

# ライブラリのインポート
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 訓練データと評価用データのcsvをそれぞれ読み込み
data_train = pd.read_csv('csv/fashion-mnist_train.csv')
data_test = pd.read_csv('csv/fashion-mnist_test.csv')

# 訓練データのデータフレームを確認
data_train.head()

# 評価データのデータフレームを確認
data_test.head()

# 訓練データのshapeを確認
data_train.shape

# 画像データのサイズとインプットのshapeを指定
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# 訓練データのデータフレームを正解ラベルのデータフレームと画像情報のデータフレームに分割
# 画像の訓練データはnumpy配列化, 正解ラベルはto_categoricalでone-hot表現に変換
X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

# 変換後の画像データの確認
X

# 変換後の正解ラベルデータの確認
y

# データセットを訓練用データとテスト用データに分割
# テストデータ : 訓練データ = 2 : 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

# 評価用データをラベルと画像に分けてnumpy配列化
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

# 
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

# それぞれの画像の0~255までの数字を0~1に収まるように調整
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

# ここまででデータの前処理終了

# CNN用のライブラリのインポート
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# バッチサイズ、ラベルのクラス数、エポック数の設定
batch_size = 256
num_classes = 10
epochs = 50

# モデルの設計
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0, 25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0, 25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])






















