# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

__all__ = ['OctConv']

from mxnet.gluon import nn


class OctConv(nn.HybridBlock):
    def __init__(self, channels, alpha_in, alpha_out, kernel_size,
                 stride=1, padding=0, groups=1, use_bias=False, depth_wise=False):
        super(OctConv, self).__init__()
        assert stride in (1, 2)
        assert alpha_in >= 0 and alpha_out >= 0
        high_weight_channels = int((1 - alpha_out) * channels)
        low_weight_channels = int(alpha_out * channels)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.stride = stride
        self.depth_wise = depth_wise
        with self.name_scope():
            self.W_HH = nn.Conv2D(high_weight_channels, kernel_size, 1, padding,
                                  groups=groups, use_bias=use_bias)
            if alpha_out != 0 and not depth_wise:
                self.W_HL = nn.Conv2D(low_weight_channels, kernel_size, 1, padding,
                                      groups=groups, use_bias=use_bias)
            if alpha_in != 0 and not depth_wise:
                self.W_LH = nn.Conv2D(high_weight_channels, kernel_size, 1, padding,
                                      groups=groups, use_bias=use_bias)
            if alpha_in != 0 and alpha_out != 0:
                self.W_LL = nn.Conv2D(low_weight_channels, kernel_size, 1, padding,
                                      groups=groups, use_bias=use_bias)

    def hybrid_forward(self, F, x_h, x_l=None):
        if self.depth_wise:
            if self.stride == 2:
                x_h = F.Pooling(data=x_h, pool_type='avg', kernel=(2, 2), stride=(2, 2))
                x_l = F.Pooling(data=x_l, pool_type='avg', kernel=(2, 2), stride=(2, 2))
            y_hh = self.W_HH(x_h)
            y_ll = self.W_LL(x_l)
            return y_hh, y_ll
        else:
            if self.stride == 2:
                x_h = F.Pooling(data=x_h, pool_type='avg', kernel=(2, 2), stride=(2, 2))
            y_hh = self.W_HH(x_h)
            if self.alpha_in == self.alpha_out == 0:
                return y_hh, None
            if self.alpha_in == 0:
                y_hl = self.W_HL(F.Pooling(data=x_h, pool_type='avg', kernel=(2, 2), stride=(2, 2)))
                return y_hh, y_hl

            y_lh = F.UpSampling(self.W_LH(x_l), scale=2, sample_type='nearest',
                                num_args=1) if self.stride == 1 else self.W_LH(x_l)
            y_h_out = y_hh + y_lh

            if self.alpha_out == 0:
                return y_h_out, None

            y_hl = self.W_HL(F.Pooling(data=x_h, pool_type='avg', kernel=(2, 2), stride=(2, 2)))

            if self.stride == 2:
                x_l = F.Pooling(data=x_l, pool_type='avg', kernel=(2, 2), stride=(2, 2))
            y_ll = self.W_LL(x_l)
            y_l_out = y_hl + y_ll
            return y_h_out, y_l_out
