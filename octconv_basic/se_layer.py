# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
__all__ = ['SELayer']

from mxnet.gluon import nn


class SELayer(nn.HybridBlock):
    def __init__(self, channel, in_channel, reduction=16, **kwargs):
        super(SELayer, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.fc = nn.HybridSequential()
            with self.fc.name_scope():
                self.fc.add(nn.Conv2D(channel // reduction, kernel_size=1, in_channels=in_channel))
                self.fc.add(nn.Activation('relu'))
                self.fc.add(nn.Conv2D(channel, kernel_size=1, in_channels=channel // reduction))
                self.fc.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self.avg_pool(x)
        y = self.fc(y)
        return F.broadcast_mul(x, y)
