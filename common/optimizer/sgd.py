import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        # learning_rate
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
