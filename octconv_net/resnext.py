# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
__all__ = [
    'oct_resnext50_32x4d', 'oct_resnext101_32x4d', 'oct_resnext101_64x4d',
    'oct_se_resnext50_32x4d', 'oct_se_resnext101_32x4d', 'oct_se_resnext101_64x4d']
import math
from octconv_basic import *
from mxnet.gluon import nn


class Block(nn.HybridBlock):
    def __init__(self, channels, alpha_in, alpha_out, cardinality, bottleneck_width,
                 stride=1, downsample=False, use_se=False, **kwargs):

        super(Block, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D
        if downsample:
            self.downsample = self.bn_act_conv(channels * 4, alpha_in, alpha_out, 1, stride)
        else:
            self.downsample = None
        self.body = AdaptiveSequential()
        self.body.add(self.bn_act_conv(group_width, alpha_in, alpha_out, 1))
        alpha_in, alpha_out = check_status(alpha_in, alpha_out)
        self.body.add(self.bn_act_conv(group_width, alpha_in, alpha_out, 3, stride, 1, cardinality))
        self.body.add(self.bn_act_conv(channels * 4, alpha_in, alpha_out, 1))

    def bn_act_conv(self, channels, alpha_in, alpha_out, kernel_size,
                    stride=1, padding=0, groups=1, use_bias=False):
        layer = AdaptiveSequential()
        with layer.name_scope():
            layer.add(fs_bn(alpha_in))
            layer.add(fs_activation('relu'))
            layer.add(OctConv(channels, alpha_in, alpha_out, kernel_size,
                              stride, padding, groups, use_bias))
        return layer

    def hybrid_forward(self, F, x_h, x_l=None):
        r_h, r_l = x_h, x_l
        x_h, x_l = self.body(x_h, x_l)
        if self.downsample:
            r_h, r_l = self.downsample(r_h, r_l)
        y_h, y_l = x_h + r_h, None if x_l is None and r_l is None else x_l + r_l
        if y_l is None:
            return y_h
        else:
            return y_h, y_l


class ResNext(nn.HybridBlock):
    def __init__(self, alpha, layers, cardinality, bottleneck_width,
                 classes=1000, use_se=False, **kwargs):
        super(ResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = 64
        with self.name_scope():
            self.features = AdaptiveSequential()
            self.features.add(nn.BatchNorm(center=False, scale=False))
            self.features.add(nn.Conv2D(channels, 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(alpha, num_layer, channels,
                                                   stride, i + 1))
                channels *= 2
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, alpha, layers, channels, stride, stage_index,
                    use_se=False):
        layer = AdaptiveSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(Block(channels,
                            alpha if stage_index != 1 else 0,
                            alpha if stage_index != 4 else 0,
                            self.cardinality, self.bottleneck_width,
                            stride, True))
            for _ in range(layers - 1):
                layer.add(Block(channels,
                                alpha if stage_index != 4 else 0,
                                alpha if stage_index != 4 else 0,
                                self.cardinality, self.bottleneck_width))
        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


resnext_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3]}


def get_resnext(alpha, num_layers, cardinality=32, bottleneck_width=4, use_se=False,
                **kwargs):
    assert num_layers in resnext_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnext_spec.keys()))
    layers = resnext_spec[num_layers]
    net = ResNext(alpha, layers, cardinality, bottleneck_width, use_se=use_se, **kwargs)

    return net


def oct_resnext50_32x4d(alpha, **kwargs):
    return get_resnext(alpha, 50, 32, 4, **kwargs)


def oct_resnext101_32x4d(alpha, **kwargs):
    return get_resnext(alpha, 101, 32, 4, **kwargs)


def oct_resnext101_64x4d(alpha, **kwargs):
    return get_resnext(alpha, 101, 64, 4, **kwargs)


def oct_se_resnext50_32x4d(alpha, **kwargs):
    return get_resnext(alpha, 50, 32, 4, **kwargs)


def oct_se_resnext101_32x4d(alpha, **kwargs):
    return get_resnext(alpha, 101, 32, 4, **kwargs)


def oct_se_resnext101_64x4d(alpha, **kwargs):
    return get_resnext(alpha, 101, 64, 4, **kwargs)
