# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
__all__ = ['get_model']
from .resnet import *
from .resnext import *
from .mobilenet import *

_src = {
    'mobilenetv3_large': mobilenetv3_large,
    'mobilenetv3_small': mobilenetv3_small,
    'oct_resnet50v1': oct_resnet50_v1,
    'oct_resnet101v1': oct_resnet101_v1,
    'oct_resnet152v1': oct_resnet152_v1,
    'oct_resnet50v2': oct_resnet50_v2,
    'oct_resnet101v2': oct_resnet101_v2,
    'oct_resnet152v2': oct_resnet152_v2,
    'oct_resnext50_32x4d': oct_resnext50_32x4d,
    'oct_resnext101_32x4d': oct_resnext101_32x4d,
    'oct_resnext101_64x4d': oct_resnext101_64x4d,
    'oct_se_resnext50_32x4d': oct_se_resnext50_32x4d,
    'oct_se_resnext101_32x4d': oct_se_resnext101_32x4d,
    'oct_se_resnext101_64x4d': oct_se_resnext101_64x4d
}


def get_model(model_name, alpha=0, **kwargs):
    print(model_name)
    assert model_name in _src.keys()
    return _src[model_name](alpha, **kwargs) if alpha > 0 else _src[model_name](**kwargs)
