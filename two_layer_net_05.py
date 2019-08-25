# -*- coding: utf-8 -*-
# 2019/08/25

import sys, os
sys.path.append(os.pardir)
import numpy as np
# /home/iwase/Documents/research/DataScience/DeepLearning/common/
# のlayers.pyから利用する。
from common.layers import *
# from common.functions import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 重みの初期化
    # ディクショナリ型の変数。以下のように順番にレイヤを追加していく。
    self.params = {}
    # 1層目の重み。ランダムな値を入れている。毎回結果は異なる。W2も同様。
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    # 1層目のバイアス。ここではバイアスは0にしている。b2も同様。
    self.params['b1'] = np.zeros(hidden_size)
    # 2層目の重み
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    # 2層目のバイアス
    self.params['b2'] = np.zeros(output_size)

    # レイヤの生成
    # 順序付き辞書。レイヤをOrderedDict()で保持する。ディクショナリに追加した要素の順番にデータを保持する。
    # 順伝播では、追加した順にレイヤのforward()メソッドを呼び出すだけで処理が完了する。
    self.layers = OrderedDict()
    # Affineクラス（layers.py）を利用してレイヤ生成
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    # Reluクラス（layers.py）を利用して活性化関数Reluレイヤを生成
    # 活性化関数：ステップ or シグモイド or ReLU
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    # 誤差もここに定義する
    # 分類問題なので、ソフトマックス関数を使った。
    # 回帰であれば恒等関数を用いるればいいはずなので、これはいらない（？）
    self.lastLayer = SoftmaxWithLoss()
    
  # 推論を行う。xはここでは画像データ
  def predict(self, x):
    # W1, W2 = self.params['W1'], self.params['W2']
    # b1, b2 = self.params['b1'], self.params['b2']

    # a1 = np.dot(x, W1) + b1
    # z1 = sigmoid(a1) # 活性化関数：シグモイド関数
    # a2 = np.dot(z1, W2) + b2
    # y = softmax(a2)

    # layersは順序付き辞書。layers.values()にはodict_value(リスト)が入っている
    for layer in self.layers.values():
      # 各クラスのforward(x)が選択されて実行される
      x = layer.forward(x)

    return x

  # 損失関数の値を求める。x:入力データ t:教師データ
  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)
    # return cross_entropy_error(y, t)

  # 認識精度を求める。
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1) # yが最大となるインデックスを取得。推論されたyを決定する。このときのyが計算された０〜９の数字
    t = np.argmax(t, axis=1) # tが最大のインデックスを取得。正解ラベル

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  # 数値微分によって勾配を求める。
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads
  
  # 勾配を求める。誤差逆伝播法。こちらのほうが数値微分による勾配計算よりも高速！！！
  def gradient(self, x, t):
    # forward
    self.loss(x, t)

    # backward
    dout = 1
    dout = self.lastLayer.backward(dout) # SoftmaxWithLoss() の最後のレイヤ

    layers = list(self.layers.values())
    layers.reverse() # リストを反転させる
    for layer in layers:
      dout = layer.backward(dout)
    
    # 設定
    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db

    return grads

'''
      W1, W2 = self.params['W1'], self.params['W2']
      b1, b2 = self.params['b1'], self.params['b2']
      grads = {}
      
      batch_num = x.shape[0]
      
      # forward
      a1 = np.dot(x, W1) + b1
      z1 = sigmoid(a1)
      a2 = np.dot(z1, W2) + b2
      y = softmax(a2)
      
      # backward
      dy = (y - t) / batch_num
      grads['W2'] = np.dot(z1.T, dy)
      grads['b2'] = np.sum(dy, axis=0)
      
      dz1 = np.dot(dy, W2.T)
      da1 = sigmoid_grad(a1) * dz1
      grads['W1'] = np.dot(x.T, da1)
      grads['b1'] = np.sum(da1, axis=0)

      return grads
'''
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

# x = np.random.rand(100, 784) # ダミーの入力データ(100枚分)
# y = net.predict(x) # ダミーの正解ラベル（100枚分)
# # print(y)

# t = np.random.rand(100, 10) # ダミーの正解ラベル（100枚分)

# grads = net.numerical_gradient(x, t)

# print(grads)
