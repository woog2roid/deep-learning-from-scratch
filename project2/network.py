"""
optimizer/weight_init comparison에서 사용할 수 있도록
2층의 layer를 사용: (sigmoid / relu) + (softmax with loss)
"""

# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


import numpy as np
from collections import OrderedDict

from common.layer.affine import *
from common.layer.relu import *
from common.layer.sigmoid import *
from common.layer.softmax_with_loss import *


class Network:
    """
    Parameters

    input_size : 입력 크기(MNIST의 경우엔 784)
    output_size : 출력 크기(MNIST의 경우엔 10)
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트(e.g. [100, 100, 100])
    activation_function : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std :
        가중치의 표준편차 지정(e.g. 0.01)
        'he'로 지정하면 'He 초깃값'으로, 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size_list,
        activation_function,
        weight_init_std,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층 생성
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            self.layers["Activation_function" + str(idx)] = activation_layer[
                activation_function
            ]()

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(
            self.params["W" + str(idx)], self.params["b" + str(idx)]
        )

        self.loss_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if str(weight_init_std).lower() in ("he"):
                # ReLU를 사용할 때의 권장 초깃값(He 초깃값)
                # 앞의 노드의 갯수의 제곱근의 두배
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ("xavier"):
                # sigmoid를 사용할 때의 권장 초깃값(Xavier 초깃값)
                # 앞의 노드의 갯수의 제곱근
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            else:
                scale = float(weight_init_std)
            
            self.params["W" + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1],
                all_size_list[idx],
            )
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = self.loss_layer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = self.layers["Affine" + str(idx)].dW
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

        return grads
