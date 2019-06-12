# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['AdaptiveSequential']
from mxnet.gluon import nn


class AdaptiveSequential(nn.HybridSequential):
    def __init__(self, prefix=None, params=None):
        super(AdaptiveSequential, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, *x):
        for block in self._children.values():
            if type(x) == tuple:
                x = block(*x)
            else:
                x = block(x)
        return x
