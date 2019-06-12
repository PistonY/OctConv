__all__ = ['check_status', 'fs_bn', 'fs_activation']

from mxnet.gluon import nn


def check_status(alpha_in, alpha_out):
    alpha_in = alpha_out if alpha_in == 0 else alpha_in
    alpha_in = 0 if alpha_out == 0 else alpha_in
    return alpha_in, alpha_out


class fs_bn(nn.HybridBlock):
    def __init__(self, alpha, **kwargs):
        super(fs_bn, self).__init__(**kwargs)
        self.h_bn = nn.BatchNorm()
        self.l_bn = nn.BatchNorm() if alpha != 0 else None

    def hybrid_forward(self, F, x_h, x_l=None):
        y_h = self.h_bn(x_h)
        y_l = None if x_l is None else self.l_bn(x_l)
        return y_h, y_l


class fs_activation(nn.Activation):
    def __init__(self, activation, **kwargs):
        super(fs_activation, self).__init__(activation, **kwargs)

    def hybrid_forward(self, F, x1, x2=None):
        x1 = F.Activation(x1, act_type=self._act_type, name='fwd_h')
        x2 = None if x2 is None else F.Activation(x2, act_type=self._act_type, name='fwd_l')
        return x1, x2
