# import torch
# from torch import nn

import paddle
import paddle.fluid as fluid


LAYER_BUILDER_DICT=dict()


def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False, num_segments=4):
    id = info['id']
    name = id

    attr = info['attrs'] if 'attrs' in info else dict()
    if 'kernel_d' in attr.keys():
        if isinstance(attr["kernel_d"], str):
            div_num = int(attr["kernel_d"].split("/")[-1])
            attr['kernel_d'] = int(num_segments / div_num)

    out, op, in_vars = parse_expr(info['expr'])
    assert(len(out) == 1)
    assert(len(in_vars) == 1)
    mod = LAYER_BUILDER_DICT[op](attr, channels, name)

    return id, out[0], mod, in_vars[0]


def build_conv(attr, channels, name):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_h'], attr['kernel_w'])
    if 'pad' in attr or 'pad_w' in attr and 'pad_h' in attr:
        padding = attr['pad'] if 'pad' in attr else (attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if 'stride' in attr or 'stride_w' in attr and 'stride_h' in attr:
        stride = attr['stride'] if 'stride' in attr else (attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    # conv = nn.Conv2d(channels, out_channels, ks, stride, padding, bias=name)
    # conv = fluid.dygraph.Conv2d(channels, out_channels, ks, stride, padding, bias=conv_bias)
    conv = fluid.layers.conv2d(channels, out_channels, ks, stride, padding, name=name)

    # return conv, out_channels
    return conv


def build_pooling(attr, channels, name):
    method = attr['mode']
    pad = attr['pad'] if 'pad' in attr else 0
    if method == 'max':
        # pool = nn.MaxPool2d(attr['kernel_size'], attr['stride'], pad,
        #                     ceil_mode=True) # all Caffe pooling use ceil model
        pool = fluid.layers.pool2d(channels, attr['kernel_size'], pool_stride=attr['stride'],
                                    pool_padding=pad, ceil_mode=True, name=name)
    elif method == 'ave':
        # pool = nn.AvgPool2d(attr['kernel_size'], attr['stride'], pad,
                            # ceil_mode=True)  # all Caffe pooling use ceil model
        pool = fluid.layers.pool2d(channels, attr['kernel_size'], 'avg', pool_stride=attr['stride'],
                                    pool_padding=pad, ceil_mode=True, name=name)
    else:
        raise ValueError("Unknown pooling method: {}".format(method))

    return pool


def build_relu(attr, channels, name):
    # return nn.ReLU(inplace=True), channels
    return fluid.layers.relu(channels, name=name)


def build_bn(attr, channels, name):
    # return nn.BatchNorm2d(channels, momentum=0.1), channels
    return fluid.layers.batch_norm(channels, name=name)


def build_linear(attr, channels, name):
    # return nn.Linear(channels, attr['num_output']), channels
    return fluid.layers.fc(channels, attr['num_output'], name=name)


def build_dropout(attr, channels, name):
    # return nn.Dropout(p=attr['dropout_ratio']), channels
    return fluid.layers.dropout(channels, attr['dropout_ratio'], name=name)

def build_conv3d(attr, channels, name):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    # conv = nn.Conv3d(channels, out_channels, ks, stride, padding, bias=conv_bias)
    conv = fluid.layers.conv3d(channels, out_channels, ks, stride, padding, name=name)

    # return conv, out_channels
    return conv

def build_pooling3d(attr, channels, name):
    method = attr['mode']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1
    if method == 'max':
        # pool = nn.MaxPool3d(ks, stride, padding,
                            # ceil_mode=True) # all Caffe pooling use ceil model
        pool_type = 'max'
    elif method == 'ave':
        # pool = nn.AvgPool3d(ks, stride, padding,
                            # ceil_mode=True)  # all Caffe pooling use ceil model
        pool_type = 'avg'
    else:
        raise ValueError("Unknown pooling method: {}".format(method))

    pool = fluid.layers.pool3d(channels, ks, pool_type, stride, padding, ceil_mode=True, name=name)

    # return pool, channels
    return pool

def build_bn3d(attr, channels, name):
    # return nn.BatchNorm3d(channels, momentum=0.1), channels
    return fluid.layers.batch_norm(channels, name=name)


LAYER_BUILDER_DICT['Convolution'] = build_conv

LAYER_BUILDER_DICT['Pooling'] = build_pooling

LAYER_BUILDER_DICT['ReLU'] = build_relu

LAYER_BUILDER_DICT['Dropout'] = build_dropout

LAYER_BUILDER_DICT['BN'] = build_bn

LAYER_BUILDER_DICT['InnerProduct'] = build_linear

LAYER_BUILDER_DICT['Conv3d'] = build_conv3d

LAYER_BUILDER_DICT['Pooling3d'] = build_pooling3d

LAYER_BUILDER_DICT['BN3d'] = build_bn3d

