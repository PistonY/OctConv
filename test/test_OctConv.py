# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
import mxnet as mx
from mxnet import nd
from octconv_basic import *

# test basic
ctx = mx.gpu(0)
oct_ker = OctConv(64, 0, 0, 3, 2, padding=1)
oct_ker.initialize(ctx=ctx)
x_in_H = nd.random.normal(shape=(1, 3, 224, 224), ctx=ctx)
x_in_L = nd.random.normal(shape=(1, 56, 56, 56), ctx=ctx)
out_H, out_L = oct_ker(x_in_H), x_in_L
print(out_H.shape, out_L.shape)
# test first layer
pic = nd.random.normal(shape=(1, 3, 224, 224), ctx=ctx)
first_layer = OctConv(64, alpha_in=0, alpha_out=0.125)
first_layer.initialize(ctx=ctx)
first_out_H, first_out_L = first_layer(pic)
print(first_out_H.shape, first_out_L.shape)
# test last layer
last_layer = OctConv(64, alpha_in=0.125, alpha_out=0)
last_layer.initialize(ctx=ctx)
output = last_layer(first_out_H, first_out_L)
print(output.shape)
# test HybridSequential
seq = AdaptiveSequential()
seq.add(OctConv(128, 0.125, 0.125))
seq.add(OctConv(128, 0.125, 0.125))
seq.initialize(ctx=ctx)
seq_out = seq(first_out_H, first_out_L)
print(seq_out[0].shape, seq_out[1].shape)
# output
# (1, 56, 112, 112) (1, 8, 56, 56)
# (1, 56, 224, 224) (1, 8, 112, 112)
# (1, 64, 224, 224)
# (1, 112, 224, 224) (1, 16, 112, 112)
