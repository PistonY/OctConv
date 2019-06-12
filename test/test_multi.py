# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

from octconv_basic.AdaptiveSequential import *
from mxnet.gluon import nn
from mxnet import nd


class n_to_n(nn.HybridBlock):
    def __init__(self):
        super(n_to_n, self).__init__()
        self.conv1 = nn.Conv2D(3, 1, 1, use_bias=False)
        self.conv2 = nn.Conv2D(3, 1, 1, use_bias=False)

    def hybrid_forward(self, F, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1, y2


class n_to_one(nn.HybridBlock):
    def __init__(self):
        super(n_to_one, self).__init__()
        self.conv1 = nn.Conv2D(3, 1, 1, use_bias=False)
        self.conv2 = nn.Conv2D(3, 1, 1, use_bias=False)

    def hybrid_forward(self, F, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1 + y2


class one_to_n(nn.HybridBlock):
    def __init__(self):
        super(one_to_n, self).__init__()
        self.conv1 = nn.Conv2D(3, 1, 1, use_bias=False)

    def hybrid_forward(self, F, x1):
        y1 = self.conv1(x1)
        return y1, y1


class one_to_one(nn.HybridBlock):
    def __init__(self):
        super(one_to_one, self).__init__()
        self.conv1 = nn.Conv2D(3, 1, 1, use_bias=False)

    def hybrid_forward(self, F, x1):
        y1 = self.conv1(x1)
        return y1


if __name__ == '__main__':
    import mxnet as mx

    ctx = mx.gpu(0)
    seq = AdaptiveSequential()
    seq.add(one_to_n(), n_to_n(), n_to_one(), one_to_one())
    seq.initialize(ctx=ctx)
    data1 = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
    data2 = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
    output = seq(data1, data2)
    print(output[0].shape)
    # (1, 16, 32, 32) (1, 16, 32, 32)
