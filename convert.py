import argparse

import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp

# python convert.py  --tf_path ../simclr/tf2/cifar10_models/firsttry_real/ckpt-780
# python convert.py  --tf_path ../simclr/tf2/cifar10_models/firsttry_real50/ckpt-1 --depth 50 --width 1 --sk_ratio 0.0 --channels_in 2048 --num_layers 2 --out_dim 64
from resnet import get_resnet, get_head
from pdb import set_trace as pb
import re

parser = argparse.ArgumentParser(description='SimCLR converter')
parser.add_argument('--tf_path', type=str, help='path of the input tensorflow file (ex: model.ckpt-250228)')
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--width', type=int, default=1)
parser.add_argument('--sk_ratio', type=float, default=0.0)
parser.add_argument('--channels_in', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--out_dim', type=int, default=64)
args = parser.parse_args()

def main():
    prefix = 'resnet_model/'
    head_prefix = 'projection_head/'

    tf_variables = cp.list_variables(args.tf_path)
    # tf_values = cp.load_variables(args.tf_path)

    # 1. read tensorflow weight into a python dict
    vars_list = []
    contrastive_vars = []
    for v in tf_variables:
        if prefix in v[0] and not v[0].endswith('/Momentum/.ATTRIBUTES/VARIABLE_VALUE'):
            vars_list.append(v[0])
        elif v[0] in {'head_supervised/linear_layer/dense/bias', 'head_supervised/linear_layer/dense/kernel'}:
            vars_list.append(v[0])
        elif head_prefix in v[0] and not v[0].endswith('/Momentum/.ATTRIBUTES/VARIABLE_VALUE'):
            contrastive_vars.append(v[0])

    sd = {}
    # ckpt_reader = tf.train.load_checkpoint(args.tf_path)
    for v in vars_list:
        variable = cp.load_variable(args.tf_path, v)
        v = re.sub('/.ATTRIBUTES/VARIABLE_VALUE', '', v)
        sd[v] = variable

    # 2. convert the state_dict to PyTorch format
    conv_keys = [k for k in sd.keys() if '/conv2d/' in k]
    conv_keys = [re.sub('/kernel', '', x) for x in conv_keys]
    conv_keys.sort()
    bn_keys = [k for k in sd.keys() if '/bn/' in k]
    bn_keys = [re.sub('/gamma|/beta|/moving_mean|/moving_variance', '', x) for x in bn_keys]
    bn_keys = list(set(bn_keys))
    bn_keys.sort()

    # depth, width, sk_ratio = name_to_params(args.tf_path)
    depth = args.depth#18
    width = args.width#1
    sk_ratio = args.sk_ratio#0

    channels_in=args.channels_in#512
    num_layers=args.num_layers#2
    out_dim=args.out_dim#64

    model = get_resnet(depth, width, sk_ratio, cifar_stem=True)
    head = get_head(channels_in, num_layers, out_dim)

    conv_op = []
    conv_op_names = []
    bn_op = []
    bn_op_names = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_op.append(m)
            conv_op_names.append(n)
        elif isinstance(m, nn.BatchNorm2d):
            bn_op.append(m)
            bn_op_names.append(n)
    assert len(vars_list) == (len(conv_op) + len(bn_op) * 4 )  # 2 for fc

    # first block
    # initial_conv_relu_max_pool = net.0
    # block_groups/j/conv2d_bn_layers/i = net.j.blocks.i.net
    # block_groups/j/shortcut_layers/i = net.j.blocks.0.projection.shortcut

    def use_key_conv(m, key):
        w = torch.from_numpy(sd[key+'/kernel']).permute(3, 2, 0, 1)
        m.weight.data = w

    def use_key_linear(m, key):
        w = torch.from_numpy(sd[key+'/kernel']).T
        m.weight.data = w

    def use_key_bn(m, key):
        gamma = torch.from_numpy(sd[key + '/gamma'])
        m.weight.data = gamma
        if (m.bias is not None):
            m.bias.data = torch.from_numpy(sd[key + '/beta'])
        m.running_mean = torch.from_numpy(sd[key + '/moving_mean'])
        m.running_var = torch.from_numpy(sd[key + '/moving_variance'])
        

    # setting up stem.... done!
    key = 'model/resnet_model/initial_conv_relu_max_pool/0/conv2d'
    use_key_conv(model.net[0].conv1, key)

    key = 'model/resnet_model/initial_conv_relu_max_pool/2/bn'
    use_key_bn(model.net[0].bn1.bn, key)

    if args.depth == 18:
        for i in range(4):
            layer_keys = [x for x in conv_keys if 'block_groups/'+str(i)+'/layers' in x]
            layer_shortcut_key = [x for x in layer_keys if 'shortcut_layers' in x]
            layer_conv_key = [x for x in layer_keys if 'conv2d_bn_layers' in x]
            layer_keys_bn = [x for x in bn_keys if 'block_groups/'+str(i)+'/layers' in x]
            layer_shortcut_key_bn = [x for x in layer_keys_bn if 'shortcut_layers' in x]
            layer_conv_key_bn = [x for x in layer_keys_bn if 'conv2d_bn_layers' in x]
            pb()

            use_key_conv(model.net[i+1].blocks[0].projection.shortcut.conv, layer_shortcut_key[0])
            use_key_bn(model.net[i+1].blocks[0].projection.bn.bn, layer_shortcut_key_bn[0])
            j_i = 0
            for j in range(len(model.net[i+1].blocks)):
                use_key_conv(model.net[i+1].blocks[j].net.conv1, layer_conv_key[j_i])
                use_key_conv(model.net[i+1].blocks[j].net.conv2, layer_conv_key[j_i+1])
                use_key_bn(model.net[i+1].blocks[j].net.bn1.bn, layer_conv_key_bn[j_i])
                use_key_bn(model.net[i+1].blocks[j].net.bn2.bn, layer_conv_key_bn[j_i+1])
                j_i+=2
    elif args.depth == 50:
        for i in range(4):
            layer_keys = [x for x in conv_keys if 'block_groups/'+str(i)+'/layers' in x]
            layer_shortcut_key = [x for x in layer_keys if 'projection_layers' in x]
            layer_conv_key = [x for x in layer_keys if 'conv_relu_dropblock_layers' in x]
            layer_keys_bn = [x for x in bn_keys if 'block_groups/'+str(i)+'/layers' in x]
            layer_shortcut_key_bn = [x for x in layer_keys_bn if 'projection_layers' in x]
            layer_conv_key_bn = [x for x in layer_keys_bn if 'conv_relu_dropblock_layers' in x]

            use_key_conv(model.net[i+1].blocks[0].projection.shortcut.conv, layer_shortcut_key[0])
            use_key_bn(model.net[i+1].blocks[0].projection.bn.bn, layer_shortcut_key_bn[0])
            j_i = 0
            for j in range(len(model.net[i+1].blocks)):
                use_key_conv(model.net[i+1].blocks[j].net.conv1, layer_conv_key[j_i])
                use_key_conv(model.net[i+1].blocks[j].net.conv2, layer_conv_key[j_i+1])
                use_key_bn(model.net[i+1].blocks[j].net.bn1.bn, layer_conv_key_bn[j_i])
                use_key_bn(model.net[i+1].blocks[j].net.bn2.bn, layer_conv_key_bn[j_i+1])
                j_i+=2
    
    # ===========================================================
    sd = {}
    # ckpt_reader = tf.train.load_checkpoint(args.tf_path)
    for v in contrastive_vars:
        variable = cp.load_variable(args.tf_path, v)
        v = re.sub('/.ATTRIBUTES/VARIABLE_VALUE', '', v)
        sd[v] = variable

    # 2. convert the state_dict to PyTorch format
    linear_keys = [k for k in sd.keys() if '/dense/' in k]
    linear_keys = [re.sub('/kernel', '', x) for x in linear_keys]
    linear_keys.sort()
    bn_keys = [k for k in sd.keys() if '/bn/' in k]
    bn_keys = [re.sub('/gamma|/beta|/moving_mean|/moving_variance', '', x) for x in bn_keys]
    bn_keys = list(set(bn_keys))
    bn_keys.sort()  

    for i in range(num_layers):
        use_key_linear(getattr(head.layers, 'linear'+str(i)), linear_keys[i])
        use_key_bn(getattr(head.layers, 'bn'+str(i)), bn_keys[i])

    save_location = f'r{depth}_{width}x_simclrv2.pth'
    torch.save({'resnet': model.state_dict(), 'head': head.state_dict()}, save_location)


if __name__ == '__main__':
    main()
