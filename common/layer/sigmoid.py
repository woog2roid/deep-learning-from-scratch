import numpy as np


class Sigmoid:
    def __init__(self):
        # 역전파에 이용하기 위해
        self.out = None

    def forward(self, x):
        # 시그모이드 함수 수식
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        # 시그모이드 함수의 오차 역전파 수식
        dx = dout * (1.0 - self.out) * self.out

        return dx
