import sys, os

sys.path.append(os.pardir)

import numpy as np

from common.layer.affine import *
from common.layer.relu import *
from common.layer.sigmoid import *
from common.layer.softmax_with_loss import *

from collections import OrderedDict


class Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 파라미터 / 편향 초기화 (??? 초깃값 사용)
        self.params = {}
        # [input_size, hidden_size]
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        # [hidden_size, output_size]
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        # 가장 마지막 레이어: 손실함수 계산 레이어 (Softmax With Loss)
        self.loss_layer = SoftmaxWithLoss()

        # Relu -> SoftMaxWithLoss
        # Affine1 -> Relu -> Affine2 -> SoftMaxWithLoss


    # (예측)결과값 산출
    # Softmax 레이어 레이어들을 모두 forward시킨다.
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 손실함수 산출
    # SoftmaxWithLoss에 prediction 값과 정답(테스트)레이블을 넣고 돌린다.
    def loss(self, train_batch, test_batch):
        prediction = self.predict(train_batch)
        return self.loss_layer.forward(prediction, test_batch)

    # 정확도 계산
    def accuracy(self, train, test):
        prediction = self.predict(train)
        prediction = np.argmax(prediction, axis=1)

        if test.ndim != 1:
            test = np.argmax(test, axis=1)

        accuracy = np.sum(prediction == test) / float(prediction.shape[0])
        return accuracy

    # 손실함수의 기울기 산출 (역전파)
    def gradient(self, train_batch, test_batch):
        # 우선 한번 forward를 돌리고(relu 같은 layer 때문)
        self.loss(train_batch, test_batch)

        # 마지막 (손실) 레이어 부터 backward를 돌린다.
        dout = self.loss_layer.backward()
        coppied_layers = list(self.layers.values())
        coppied_layers.reverse()
        for layer in coppied_layers:
            dout = layer.backward(dout)

        # 결과 저장
        # 각각의 layer에 변수(dW, db)로 저장되어 있게 되는 것을 묶어서 grads로 return
        # 가중치의 기울기, 편향의 기울기
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W2"], grads["b2"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads
