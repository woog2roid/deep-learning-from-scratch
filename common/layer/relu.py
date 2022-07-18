import numpy as np


class Relu:
    def __init__(self):
        # boolean 배열로 마스킹할 예정
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        # self.mask가 true인 index는 0으로
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # dout을 전달 받지 못했거나, none을 받아서 오류 발생
        # Affine에서 None을 dout으로 반환
        dout[self.mask] = 0
        dx = dout

        return dx
