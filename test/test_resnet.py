# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from octconv_basic import *
from octconv_net import *
# from gluoncv.model_zoo.resnet import

if __name__ == '__main__':
    ctx = mx.gpu()
    data = mx.random.normal(shape=(1, 3, 224, 224), ctx=ctx)
    # fir = OctConv_first(32, kernel_size=7, stride=2, padding=3, alpha_out=0.125)
    # fir.initialize(ctx=ctx)
    # fir_out1, fir_out_2 = fir(data)
    # print(fir_out1.shape, fir_out_2.shape)
    # res_block = BottleneckV2(32, alpha_in=0.125, stride=1, downsample=False)
    # res_block.initialize(ctx=ctx)
    # res_out1, res_out_2 = res_block(fir_out1, fir_out_2)
    # print(res_out1.shape, res_out_2.shape)
    # last_layer = OctConv_last(32, 0.125)
    # # last_layer.add(OctConv_last(32, 0.125))
    # #                nn.BatchNorm(),
    # #                nn.Activation('relu'))
    # last_layer.initialize(ctx=ctx)
    # output = last_layer(res_out1, res_out_2)
    # print(output.shape)
    # model = OctResNetV1(0.125, [OctBottleneckV1, BottleneckV1], [3 - 1, 4, 6 - 1, 3], [64, 256, 512, 1024, 2048])
    # model = OctResNetV2(0.125, [OctBottleneckV2, BottleneckV2], [3 - 1, 4, 6 - 1, 3], [64, 256, 512, 1024, 2048])
    # model = oct_resnet101_v2(0.125)
    model = oct_resnet50_v2(0.125)
    model.initialize(ctx=ctx)
    model.summary(data)
    # model.summary(data)
    # out = model(data)
    # print(out.shape)
    # output = model(data)
    # print(output.shape)
    # data = mx.random.normal(shape=(1, 3, 7, 7), ctx=ctx)
    # avg_pool = nd.Pooling(data, (2, 2), 'avg', stride=(2, 2))
    # print(avg_pool.shape)
    # up = nd.UpSampling(avg_pool, scale=2, sample_type='nearest', num_args=1)
    # print(up.shape)

    digraph = mx.viz.plot_network(model(mx.sym.var("data", shape=(1, 3, 224, 224))),  save_format='png')
    digraph.render(filename='oct_resnet50_v2')
