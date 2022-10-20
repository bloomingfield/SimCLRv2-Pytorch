# https://github.com/Separius/SimCLRv2-Pytorch
# https://github.com/google-research/simclr

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from pdb import set_trace as pb

BATCH_NORM_EPSILON = 1e-5

# variance_scaling_initializer
# https://docs.w3cub.com/tensorflow~python/tf/variance_scaling_initializer
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/init_ops.py
# https://github.com/keras-team/keras/blob/v2.10.0/keras/initializers/initializers_v2.py#L522-L672
def variance_scaling_initializer(tensor):
    scale=1.0
    mode='fan_in'
    distribution='normal'
    with torch.no_grad():
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale /= max(1., fan_in)
        stddev = math.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(tensor, 0.0, stddev, -2*stddev, 2*stddev)
        return tensor


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True, init_zero=False):
        batchnorm = nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON)
        if init_zero:
            nn.init.constant_(batchnorm.weight,0)
            nn.init.constant_(batchnorm.bias, 0)
        else:
            nn.init.constant_(batchnorm.weight,1)
            nn.init.constant_(batchnorm.bias, 0)
        super().__init__(OrderedDict([('bn', batchnorm), ('relu', nn.ReLU() if relu else nn.Identity())]))

def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding = (kernel_size - 1) // 2, bias=bias)
    # padding = (kernel_size - 1) // 2
    # padding=('SAME' if stride == 1 else 'VALID')
    variance_scaling_initializer(convolution.weight)
    # torch.nn.init.kaiming_uniform_(convolution.weight, mode='fan_out', nonlinearity='relu')
    # nn.init.kaiming_normal_(convolution.weight, mode='fan_out', nonlinearity='relu')
    if convolution.bias is not None:
        torch.nn.init.zeros_(convolution.bias)
    return convolution


class SelectiveKernel(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(conv(in_channels, 2 * out_channels, stride=stride),
                                       BatchNormRelu(2 * out_channels))
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(conv(out_channels, mid_dim, kernel_size=1), BatchNormRelu(mid_dim),
                                         conv(mid_dim, 2 * out_channels, kernel_size=1))

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(OrderedDict([
                                          ('zero_pad', nn.ZeroPad2d((0, 1, 0, 1))),
                                          # kernel_size = 2 => padding = 1
                                          ('avgpool2d', nn.AvgPool2d(kernel_size=2, stride=stride, padding=0)),
                                          ('conv', conv(in_channels, out_channels, kernel_size=1))
                                          ]))
        else:
            self.shortcut = nn.Sequential(OrderedDict([
                                          ('conv', conv(in_channels, out_channels, kernel_size=1, stride=stride))
                                          ]))
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [
               ('conv1', conv(in_channels, out_channels, kernel_size=3, stride=stride)),
               ('bn1', BatchNormRelu(out_channels))
            ]

        ops.append(('conv2', conv(out_channels, out_channels, kernel_size=3, stride=1)))
        ops.append(('bn2', BatchNormRelu(out_channels, relu=False, init_zero=True)))
        self.net = nn.Sequential(OrderedDict(ops))

    def forward(self, x):
        # pb()
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))



class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [
                ('conv1', conv(in_channels, out_channels, kernel_size=1)), 
                ('bn1', BatchNormRelu(out_channels))
            ]
        if sk_ratio > 0:
            ops.append(('sk', SelectiveKernel(out_channels, out_channels, stride, sk_ratio)))
        else:
            ops.append(('conv2', conv(out_channels, out_channels, stride=stride)))
            ops.append(('bn2', BatchNormRelu(out_channels)))
        ops.append(('conv3', conv(out_channels, out_channels * 4, kernel_size=1)))
        ops.append(('bn3', BatchNormRelu(out_channels * 4, relu=False, init_zero=True)))
        self.net = nn.Sequential(OrderedDict(ops))

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, block, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList([block(in_channels, out_channels, stride, sk_ratio, True)])
        self.channels_out = out_channels * block.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(block(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

# https://github.com/google-research/simclr/blob/81e63a27be8b400a268cd4e1710af7590d172a36/resnet.py#L567
class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier, cifar_stem):
        ops = []
        if cifar_stem:
            channels = 64 * width_multiplier
            ops.append(('conv1', conv(3, channels, stride=1)))
            ops.append(('bn1', BatchNormRelu(channels)))
        else:
            channels = 64 * width_multiplier // 2
            if sk_ratio > 0:
                ops.append(('conv1', conv(3, channels, stride=2)))
                ops.append(('bn1', BatchNormRelu(channels)))
                ops.append(('conv2', conv(channels, channels)))
                ops.append(('bn2', BatchNormRelu(channels)))
                ops.append(('conv3', conv(channels, channels * 2)))
            else:
                ops.append(('conv1', conv(3, channels * 2, kernel_size=7, stride=2)))
            ops.append(('bn3', BatchNormRelu(channels * 2)))
            ops.append(('maxpool2d', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
        super().__init__(OrderedDict(ops))

class ResNetMini(nn.Module):
    def __init__(self, block, layers, width_multiplier, sk_ratio, cifar_stem):
        super().__init__()
        ops = []
        channels = 64 * width_multiplier
        ops.append(('conv1', conv(3, channels, stride=1)))
        ops.append(('bn1', BatchNormRelu(channels)))
        self.net = nn.Sequential(OrderedDict(ops))

    def forward(self, x, apply_fc=False):
        # h = self.net(x)
        # avg = nn.AdaptiveAvgPool2d((1, 1)) # this is equivalent to below
        # h2 = avg(h).squeeze()
        h = self.net(x).mean(dim=[2, 3])
        return h

class ResNet(nn.Module):
    def __init__(self, block, layers, width_multiplier, sk_ratio, cifar_stem):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier, cifar_stem)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(block, layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(block, layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(block, layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(block, layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        # pb()
        self.fc = nn.Linear(channels_in, 1000)


    def forward(self, x, apply_fc=False):
        # h = self.net(x)
        # avg = nn.AdaptiveAvgPool2d((1, 1)) # this is equivalent to below
        # h2 = avg(h).squeeze()
        h = self.net(x).mean(dim=[2, 3])
        if apply_fc:
            h = self.fc(h)
        return h

# =============================================================================

class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            linear = nn.Linear(channels_in, dim, bias=False)
            torch.nn.init.normal_(linear.weight, std=0.01)
            if linear.bias is not None:
                torch.nn.init.zeros_(linear.bias)
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
            # https://www.tensorflow.org/api_docs/python/tf/random_normal_initializer
            # https://pytorch.org/docs/stable/nn.init.html
            self.layers.append(('linear'+str(i), linear))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            nn.init.constant_(bn.weight,1)
            nn.init.zeros_(bn.bias)
            if i == num_layers - 1:
                bn.bias = None
            self.layers.append(('bn'+str(i), bn))
            if relu:
                self.layers.append(('relu'+str(i), nn.ReLU()))

        self.layers = nn.Sequential(OrderedDict(self.layers))

    def forward(self, x):
        xo = self.layers(x)
        return xo

def supervised_head(input_dim, output_dim=128, zero_init_logits_layer=False, **kwargs):
    """Add supervised head & also add its variables to inblock collection."""
    linear = nn.Linear(input_dim, output_dim)
    torch.nn.init.normal_(linear.weight, std=0.01)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)
    if zero_init_logits_layer:
        torch.nn.init.zeros_(linear.weight)
    linear = nn.Sequential(OrderedDict([('linear', linear)]))
    return linear

# =============================================================================

def get_resnet(depth=50, width_multiplier=1, sk_ratio=0, cifar_stem=False, batchnorm_momentum=0.1, mini=False):  # sk_ratio=0.0625 is recommended
    layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}[depth]
    if depth == 18 or depth == 34:
        block = ResidualBlock
    else:
        block = BottleneckBlock

    if mini:
        resnet = ResNetMini(block, layers, width_multiplier, sk_ratio, cifar_stem)
    else:
        resnet = ResNet(block, layers, width_multiplier, sk_ratio, cifar_stem)

    for m in resnet.modules():
        # set batchnorm momentum
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = batchnorm_momentum
    # pb()
    # from torchsummary import summary
    # summary(resnet.cuda(), (3, 32, 32))
    # https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
    return resnet

def get_head(channels_in, num_layers, out_dim, batchnorm_momentum=0.1):
    head = ContrastiveHead(channels_in, num_layers=num_layers, out_dim=out_dim)
    for m in head.modules():
        # set batchnorm momentum
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = batchnorm_momentum
    return head

# ====================================================================

def name_to_params(checkpoint):
    sk_ratio = 0.0625 if '_sk1' in checkpoint else 0
    if 'r50_' in checkpoint:
        depth = 50
    elif 'r101_' in checkpoint:
        depth = 101
    elif 'r152_' in checkpoint:
        depth = 152
    else:
        raise NotImplementedError

    if '_1x_' in checkpoint:
        width = 1
    elif '_2x_' in checkpoint:
        width = 2
    elif '_3x_' in checkpoint:
        width = 3
    else:
        raise NotImplementedError

    return depth, width, sk_ratio

# ====================================================================
class ResNetPlusLayer(nn.Module):
    def __init__(self, resnet, projection_head, layers=1):
        super().__init__()

        self.resnet = resnet
        # pb()
        self.projection_head = projection_head
        if layers == 1:
            self.projection_head.layers = self.projection_head.layers[0:3]

    def forward(self, x):
        h = self.resnet(x)
        h = self.projection_head(h)
        return h
# ====================================================================

class SimCLRv2Resnet(nn.Module):
    def __init__(self, pth_path, module='encoder'):
        super(SimCLRv2Resnet, self).__init__()
        model = get_resnet(*name_to_params(pth_path))
        head = get_head(model.channels_out)
        if module == 'encoder':
            model.load_state_dict(torch.load(pth_path)['resnet'])
            self.model = model
        elif module == 'head':
            head.load_state_dict(torch.load(pth_path)['head'])
            self.model = head
            # pb()
        elif module == 'encoder+head':
            model.load_state_dict(torch.load(pth_path)['resnet'])
            head.load_state_dict(torch.load(pth_path)['head'])
            self.model = ResNetPlusLayer(model, head)
        else:
            raise NameError('Select different module')

    def forward(self, x):
        return self.model(x)
