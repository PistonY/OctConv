# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
__all__ = [
    'oct_resnet50_v1', 'oct_resnet101_v1', 'oct_resnet152_v1',
    'oct_resnet50_v2', 'oct_resnet101_v2', 'oct_resnet152_v2',
    'get_resnet']
from octconv_basic import *
from mxnet.gluon import nn


class OctBottleneckV1(nn.HybridBlock):
    def __init__(self, channels, alpha_in, alpha_out,
                 stride=1, downsample=False, use_se=False, **kwargs):
        super(OctBottleneckV1, self).__init__(**kwargs)

        if downsample:
            self.downsample = self.conv_bn(channels, alpha_in, alpha_out, 1, stride)
        else:
            self.downsample = None

        self.body = AdaptiveSequential()
        self.body.add(self.conv_bn(channels // 4, alpha_in, alpha_out, 1, use_act=True))
        alpha_in, alpha_out = check_status(alpha_in, alpha_out)
        self.body.add(self.conv_bn(channels // 4, alpha_in, alpha_out, 3, stride, 1, use_act=True))
        self.body.add(self.conv_bn(channels, alpha_in, alpha_out, 1, use_act=True))

        self.relu = fs_activation('relu')

    def conv_bn(self, channels, alpha_in, alpha_out, kernel_size, stride=1, padding=0,
                groups=1, use_act=False):
        layer = AdaptiveSequential()
        with layer.name_scope():
            layer.add(OctConv(channels, alpha_in, alpha_out, kernel_size, stride,
                              padding, groups, False))
            layer.add(fs_bn(alpha_out))
            if use_act:
                layer.add(fs_activation('relu'))
        return layer

    def hybrid_forward(self, F, x_h, x_l=None):
        r_h, r_l = x_h, x_l
        x_h, x_l = self.body(x_h, x_l)
        if self.downsample:
            r_h, r_l = self.downsample(r_h, r_l)
        y_h, y_l = self.relu(x_h + r_h, None if x_l is None and r_l is None else x_l + r_l)
        if y_l is None:
            return y_h
        else:
            return y_h, y_l


class OctBottleneckV2(nn.HybridBlock):
    def __init__(self, channels, alpha_in, alpha_out, stride=1, downsample=False,
                 use_se=False, **kwargs):
        super(OctBottleneckV2, self).__init__(**kwargs)

        if downsample:
            self.downsample = self.bn_act_conv(channels, alpha_in, alpha_out, 1, stride)
        else:
            self.downsample = None

        self.body = AdaptiveSequential()
        self.body.add(self.bn_act_conv(channels // 4, alpha_in, alpha_out, 1))
        alpha_in, alpha_out = check_status(alpha_in, alpha_out)
        self.body.add(self.bn_act_conv(channels // 4, alpha_in, alpha_out, 3, stride, 1))
        self.body.add(self.bn_act_conv(channels, alpha_in, alpha_out, 1))

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


class OctResNetV1(nn.HybridBlock):
    def __init__(self, alpha, block, layers, channels, classes=1000, use_se=False, **kwargs):
        super(OctResNetV1, self).__init__(**kwargs)
        with self.name_scope():
            self.features = AdaptiveSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(alpha, block, num_layer, channels[i + 1],
                                                   stride, i + 1, in_channels=channels[i]))

            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, alpha, block, layers, channels, stride, stage_index,
                    in_channels=0, use_se=False):
        layer = AdaptiveSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels,
                            alpha if stage_index != 1 else 0,
                            alpha if stage_index != 4 else 0,
                            stride, channels != in_channels))
            for _ in range(layers - 1):
                layer.add(block(channels,
                                alpha if stage_index != 4 else 0,
                                alpha if stage_index != 4 else 0))
        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class OctResNetV2(nn.HybridBlock):
    def __init__(self, alpha, block, layers, channels, classes=1000, use_se=False, **kwargs):
        super(OctResNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = AdaptiveSequential()
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(alpha, block, num_layer, channels[i + 1],
                                                   stride, i + 1, in_channels=channels[i]))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, alpha, block, layers, channels, stride, stage_index,
                    in_channels=0, use_se=False):
        layer = AdaptiveSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels,
                            alpha if stage_index != 1 else 0,
                            alpha if stage_index != 4 else 0,
                            stride, channels != in_channels))
            for _ in range(layers - 1):
                layer.add(block(channels,
                                alpha if stage_index != 4 else 0,
                                alpha if stage_index != 4 else 0))
        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


resnet_spec = {18: ([2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ([3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ([3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ([3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ([3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [OctResNetV1, OctResNetV2]
resnet_block_versions = [OctBottleneckV1, OctBottleneckV2]


def get_resnet(alpha, version, num_layers, **kwargs):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))
    layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2." % version
    resnet_class = resnet_net_versions[version - 1]
    block_class = resnet_block_versions[version - 1]
    net = resnet_class(alpha, block_class, layers, channels, **kwargs)

    return net


def oct_resnet50_v1(alpha, **kwargs):
    return get_resnet(alpha, 1, 50, **kwargs)


def oct_resnet101_v1(alpha, **kwargs):
    return get_resnet(alpha, 1, 101, **kwargs)


def oct_resnet152_v1(alpha, **kwargs):
    return get_resnet(alpha, 1, 152, **kwargs)


def oct_resnet50_v2(alpha, **kwargs):
    return get_resnet(alpha, 2, 50, **kwargs)


def oct_resnet101_v2(alpha, **kwargs):
    return get_resnet(alpha, 2, 101, **kwargs)


def oct_resnet152_v2(alpha, **kwargs):
    return get_resnet(alpha, 2, 152, **kwargs)
