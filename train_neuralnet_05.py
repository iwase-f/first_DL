# -*- coding: utf-8 -*-
# 2019/08/25

import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net_05 import TwoLayerNet # 自分で入力したtwo_layer_netを使う。
# from common.gradient import gradient
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

# データの読み込み
# MNISTの形式。(訓練画像, 訓練ラベル), (テスト画像, テストラベル)
# 教師データ、テストデータ、ともに6000個のデータ
# いずれのデータもNumPy配列形式
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(type(x_train))
# print(x_train[0])
# print(t_train[0])
# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
# print(train_size)
# test_size = x_train.shape[0]
# print(test_size)
batch_size = 100

# 学習係数
learning_rate = 0.1

train_loss_list = []

# テストデータで評価用
train_acc_list = []
test_acc_list = []
# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

# 2層ニューラルネットワーク。これでまず重みの初期化、レイヤの生成を行う。
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x = []
x_epochs = []
epoch = 0
for i in range(iters_num):
  # train_sizeからランダムに選択され、長さがbatch_sizeのリストを生成
  # ここでtrain_sizeは6000までの数値、batch_sizeは100
  batch_mask = np.random.choice(train_size, batch_size) 
  # print(batch_mask)
  # print(x_train)
  # 6000個の訓練画像から100個の画像がランダムに選ばれる
  x_batch = x_train[batch_mask] 
  # 教師ラベルも同様に6000個のラベルから100個のラベルがランダムに選ばれる
  t_batch = t_train[batch_mask]

  # 勾配の計算
  # grad = network.numerical_gradient(x_batch, t_batch)
  grad = network.gradient(x_batch, t_batch) # 高速版！

  # パラメータの更新。この更新によって重みパラメータが上で求めた勾配を考慮して更新されていく。
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  # 学習経過の記録
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)
  
  # 1エポックごとに認識精度を計算
  if i % iter_per_epoch == 0:
    # 教師データを使ってパラメータを更新したので、まずは教師データできちんと精度が出ているかチェック
    train_acc = network.accuracy(x_train, t_train)
    # print(type(x_train))
    # 今度はテストデータで精度が出るかチェック！
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    # print('iteration=',i)
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    x_epochs.append(epoch)
    epoch += 1

  # 結果の確認
  # 結果
  print(network.predict(x_test)[0])
  # 正解
  print(t_test[0])

  x.append(i) # グラフ用

# 最終的なパラメータの確認
for key in ('W1', 'b1', 'W2', 'b2'):
  print(key)
  print(network.params[key])

plt.plot(x, train_loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

plt.plot(x_epochs, train_acc_list)
plt.plot(x_epochs, test_acc_list)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


# plt.pause(.01)

# plt.plot(x, train_loss)
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.show()

