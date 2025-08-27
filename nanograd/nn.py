import random

from nanograd.grad import Scalar


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Scalar(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        features = [n(x) for n in self.neurons]
        return features[0] if len(features) == 1 else features

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs

        # generally only want non-linearity on the last layer
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(sizes) - 2))
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


class SGDOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
