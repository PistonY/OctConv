# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
# v3

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']

from mxnet.gluon import nn


class h_swish(nn.HybridBlock):
    def hybrid_forward(self, F, x):
        return x * F.clip(x + 3, 0, 6) / 6


class h_sigmoid(nn.HybridBlock):
    def hybrid_forward(self, F, x):
        return F.clip(x + 3, 0, 6) / 6


class Relu(nn.HybridBlock):
    def hybrid_forward(self, F, x):
        return F.Activation(x, act_type='relu')


class SE_layer(nn.HybridBlock):
    def __init__(self, channel, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.se = nn.HybridSequential()
        with self.se.name_scope():
            self.se.add(nn.Conv2D(channel // reduction, kernel_size=1))
            self.se.add(Relu())
            self.se.add(nn.Conv2D(channel, kernel_size=1))
            self.se.add(h_sigmoid())

    def hybrid_forward(self, F, x):
        y = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        y = self.se(y)
        return F.broadcast_mul(x, y)


# can not work
# class cbam_module(nn.HybridBlock):
#     def __init__(self, channel, reduction=4, k=3, **kwargs):
#         super().__init__(**kwargs)
#         # ChannelAttention
#         self.avg_pool = nn.GlobalAvgPool2D()
#         self.max_pool = nn.GlobalMaxPool2D()
#         self.cat = nn.HybridSequential()
#         self.cat.add(
#             nn.Conv2D(channel // reduction, kernel_size=1),
#             Relu(),
#             nn.Conv2D(channel, kernel_size=1),
#         )
#         self.h_sigmoid = h_sigmoid()
#         # SpatialAttention
#         assert k in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if k == 7 else 1
#         self.sat = nn.HybridSequential()
#         self.sat.add(
#             nn.Conv2D(1, k, padding=padding, use_bias=False),
#             nn.BatchNorm(),
#             h_sigmoid(),
#         )
#
#     def hybrid_forward(self, F, x):
#         # ChannelAttention
#         avg_out = self.cat(self.avg_pool(x))
#         max_out = self.cat(self.max_pool(x))
#         x = F.broadcast_mul(x, self.h_sigmoid(avg_out + max_out))
#         # SpatialAttention
#         avg_out = F.mean(x, axis=1, keepdims=True)
#         max_out = F.max(x, axis=1, keepdims=True)
#         out = F.concat(avg_out, max_out, dim=1)
#         out = self.sat(out)
#         return F.broadcast_mul(x, out)


class BottleneckV3(nn.HybridBlock):
    def __init__(self, in_channels, exp_channels, out_channels,
                 kernel_size, stride, non_linearities, se, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.out = nn.HybridSequential()
        with self.out.name_scope():
            self.out.add(nn.Conv2D(exp_channels, 1, 1, 0, use_bias=False))
            self.out.add(nn.BatchNorm())
            self.out.add(non_linearities())
            self.out.add(nn.Conv2D(exp_channels, kernel_size, stride, kernel_size // 2,
                                   groups=exp_channels, use_bias=False))
            self.out.add(nn.BatchNorm())
            self.out.add(non_linearities())
            self.out.add(nn.Conv2D(out_channels, 1, 1, 0, use_bias=False))
            self.out.add(nn.BatchNorm())
            if se is not None:
                self.out.add(se(out_channels))

        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.HybridSequential()
            self.shortcut.add(nn.Conv2D(out_channels, 1, 1, 0, use_bias=False))
            self.shortcut.add(nn.BatchNorm())
        else:
            self.shortcut = None

    def hybrid_forward(self, F, x):
        r = x if self.shortcut is None else self.shortcut(x)
        out = self.out(x)
        out = out + r if self.stride == 1 else out
        return out


class mobilenetv3_large(nn.HybridBlock):
    def __init__(self, classes=1000, multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='feature_')
            with self.features.name_scope():
                self.features.add(
                    nn.Conv2D(16, 3, 2, 1, use_bias=False),
                    nn.BatchNorm(),
                    h_swish(),
                    BottleneckV3(16, 16, 16, 3, 1, Relu, None),
                    BottleneckV3(16, 64, 24, 3, 2, Relu, None),
                    BottleneckV3(24, 72, 24, 3, 1, Relu, None),
                    BottleneckV3(24, 72, 40, 5, 2, Relu, SE_layer),
                    BottleneckV3(40, 120, 40, 5, 1, Relu, SE_layer),
                    BottleneckV3(40, 120, 40, 5, 1, Relu, SE_layer),
                    BottleneckV3(40, 240, 80, 3, 2, h_swish, None),
                    BottleneckV3(80, 200, 80, 3, 1, h_swish, None),
                    BottleneckV3(80, 184, 80, 3, 1, h_swish, None),
                    BottleneckV3(80, 184, 80, 3, 1, h_swish, None),
                    BottleneckV3(80, 480, 112, 3, 1, h_swish, SE_layer),
                    BottleneckV3(112, 672, 112, 3, 1, h_swish, SE_layer),
                    BottleneckV3(112, 672, 160, 5, 1, h_swish, SE_layer),
                    BottleneckV3(160, 672, 160, 5, 2, h_swish, SE_layer),
                    BottleneckV3(160, 960, 160, 5, 1, h_swish, SE_layer),
                    nn.Conv2D(960, 1, 1, 0, use_bias=False),
                    nn.BatchNorm(),
                    h_swish(),
                    nn.GlobalAvgPool2D(),
                    nn.Conv2D(int(1280 * multiplier), 1, 1, 0),
                    h_swish(),
                    nn.Conv2D(classes, 1, 1, 0),
                    nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x


class mobilenetv3_small(nn.HybridBlock):
    def __init__(self, classes=1000, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='feature_')
            with self.features.name_scope():
                self.features.add(
                    nn.Conv2D(16, 3, 2, 1, use_bias=False),
                    nn.BatchNorm(),
                    h_swish(),
                    BottleneckV3(16, 16, 16, 3, 2, Relu, SE_layer),
                    BottleneckV3(16, 72, 24, 3, 2, Relu, None),
                    BottleneckV3(24, 88, 24, 3, 1, Relu, None),
                    BottleneckV3(24, 96, 40, 5, 2, Relu, SE_layer),
                    BottleneckV3(40, 240, 40, 5, 1, h_swish, SE_layer),
                    BottleneckV3(40, 240, 40, 5, 1, h_swish, SE_layer),
                    BottleneckV3(40, 120, 48, 5, 1, h_swish, SE_layer),
                    BottleneckV3(48, 144, 48, 5, 1, h_swish, SE_layer),
                    BottleneckV3(48, 288, 96, 5, 1, h_swish, SE_layer),
                    BottleneckV3(96, 576, 96, 5, 1, h_swish, SE_layer),
                    BottleneckV3(96, 576, 96, 5, 2, h_swish, SE_layer),
                    nn.Conv2D(576, 1, 1, 0, use_bias=False),
                    nn.BatchNorm(),
                    h_swish(),
                    nn.GlobalAvgPool2D(),
                    nn.Conv2D(1280, 1, 1, 0),
                    h_swish(),
                    nn.Conv2D(classes, 1, 1, 0),
                    nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x
