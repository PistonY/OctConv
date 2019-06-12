# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

from octconv_net.resnet import OctBottleneckV1, OctBottleneckV2
from mxnet import nd
import mxnet as mx

ctx = mx.gpu()
# first layer
dt_f = nd.random.normal(shape=(1, 32, 56, 56), ctx=ctx)
b1_first = OctBottleneckV1(64, 0, 0.125, downsample=True)
b1_first.initialize(ctx=ctx)
b1_f_out1, b1_f_out2 = b1_first(dt_f)
print(b1_f_out1.shape, b1_f_out2.shape)
# hidden layer
b1_hidden = OctBottleneckV1(128, 0.125, 0.125, 2, True)
b1_hidden.initialize(ctx=ctx)
b1_h_out1, b1_h_out2 = b1_hidden(b1_f_out1, b1_f_out2)
print(b1_h_out1.shape, b1_h_out2.shape)
# last layer
b1_last = OctBottleneckV1(256, 0.125, 0, stride=2, downsample=True)
b1_last.initialize(ctx=ctx)
b1_l_out1 = b1_last(b1_h_out1, b1_h_out2)
print(b1_l_out1.shape)
# output
# (1, 56, 56, 56) (1, 8, 28, 28)
# (1, 112, 28, 28) (1, 16, 14, 14)
# (1, 256, 14, 14)


b2_first = OctBottleneckV2(64, 0, 0.125, downsample=True)
b2_first.initialize(ctx=ctx)
b2_f_out1, b2_f_out2 = b2_first(dt_f)
print(b2_f_out1.shape, b2_f_out2.shape)
b2_hidden = OctBottleneckV2(64, 0.125, 0.125, 2, True)
b2_hidden.initialize(ctx=ctx)
b2_h_out1, b2_h_out2 = b2_hidden(b2_f_out1, b2_f_out2)
print(b2_h_out1.shape, b2_h_out2.shape)
b2_last = OctBottleneckV2(256, 0.125, 0, stride=2, downsample=True)
b2_last.initialize(ctx=ctx)
b2_l_out1 = b2_last(b2_h_out1, b2_h_out2)
print(b2_l_out1.shape)
# output
# (1, 56, 56, 56) (1, 8, 28, 28)
# (1, 56, 28, 28) (1, 8, 14, 14)
# (1, 256, 14, 14)

# Nice!!!!
