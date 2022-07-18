import numpy as np


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태 / ex:0100000)

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        # print(self.y, "소프트맥스 출력 값")
        self.loss = self.cross_entropy_error(self.y, self.t)
        # print(self.loss, "손실함수 값")

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

    # 이 아래로는 softmax 및 교차엔트로피오차 정의.
    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
