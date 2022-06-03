"""
Inspired by https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

(Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from numpy import prod
from typing import Iterable

from ttconv import TTConvEinsumContract, TTConvGaussian


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, option="A", **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicBlock._create_conv_layer(in_planes, planes, kernel_size, stride, **kwargs)
        self.bn1 = BasicBlock._create_bn_layer(planes)
        self.conv2 = BasicBlock._create_conv_layer(planes, planes, kernel_size, 1, **kwargs)
        self.bn2 = BasicBlock._create_bn_layer(planes)
        self.shortcut = BasicBlock._create_shortcut_layer(in_planes, planes, stride, option)
    
    @classmethod
    def _create_conv_layer(cls, in_planes, planes, kernel_size, stride, **kwargs):
        return nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False
        )
    
    @classmethod
    def _create_bn_layer(cls, planes):
        return nn.BatchNorm2d(planes)
    
    @classmethod
    def _create_shortcut_layer(cls, in_planes, planes, stride, option):
        res = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                res = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                res = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        cls.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(cls.expansion * planes),
                )
        return res

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_TT(BasicBlock):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, option="A", **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicBlock_TT._create_conv_layer(in_planes, planes, kernel_size, stride, **kwargs)
        self.bn1 = BasicBlock_TT._create_bn_layer(prod(planes))
        self.conv2 = BasicBlock_TT._create_conv_layer(planes, planes, kernel_size, 1, **kwargs)
        self.bn2 = BasicBlock_TT._create_bn_layer(prod(planes))
        self.shortcut = BasicBlock_TT._create_shortcut_layer(prod(in_planes), prod(planes), stride, option)
    
    @classmethod
    def _create_conv_layer(cls, in_c_modes, out_c_modes, kernel_size, stride, gaussian=False, **kwargs):
        conv_func = TTConvGaussian if gaussian else TTConvEinsumContract
        return conv_func(
            in_c_modes, out_c_modes, kernel_size=kernel_size,
            stride=stride, padding=kernel_size//2, **kwargs
        )


class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, inner_modes=[16, 32, 64],
                 kernel_size=3, num_classes=10, ranks=None, **kwargs):
        super(ResNet, self).__init__()
        self.cur_inner_dim = inner_modes[0]
        self.cur_inner_prod = self._update_prod()

        self.conv1 = nn.Conv2d(
            in_channels, self.cur_inner_prod, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.cur_inner_prod)

        if not isinstance(ranks, Iterable):
            ranks = [ranks] * 3
        self.layer1 = self._make_layer(block, inner_modes[0], num_blocks[0], ranks[0], kernel_size, stride=1, **kwargs)
        self.layer2 = self._make_layer(block, inner_modes[1], num_blocks[1], ranks[1], kernel_size, stride=2, **kwargs)
        self.layer3 = self._make_layer(block, inner_modes[2], num_blocks[2], ranks[2], kernel_size, stride=2, **kwargs)
        self.linear = nn.Linear(self.cur_inner_prod, num_classes)

        self.apply(_weights_init)
    
    def _update_prod(self):
        return prod(self.cur_inner_dim) if isinstance(self.cur_inner_dim, Iterable) else self.cur_inner_dim

    def _make_layer(self, block, c, num_blocks, ranks, kernel_size, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if block is BasicBlock:
                new_layer = block(self.cur_inner_prod, prod(c), kernel_size=kernel_size, stride=stride)
            elif block is BasicBlock_TT:
                new_layer = block(self.cur_inner_dim, c, kernel_size=kernel_size, stride=stride, ranks=ranks, **kwargs)
            else:
                raise ValueError('Unknown block')
            
            layers.append(new_layer)
            self.cur_inner_dim = c
            self.cur_inner_prod = self._update_prod()

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Img_CIFARResNet(num_blocks, tt, **kwargs):
    block = BasicBlock_TT if tt else BasicBlock
    inner_modes = [(2, 4, 2), (2, 4, 4), (4, 4, 4)] if tt else [16, 32, 64]
    
    return ResNet(3, block, [num_blocks] * 3, inner_modes=inner_modes, **kwargs)


def Img_CIFARResNet20(**kwargs): return Img_CIFARResNet(3, tt=False, **kwargs)
def Img_CIFARResNet32(**kwargs): return Img_CIFARResNet(5, tt=False, **kwargs)
def Img_CIFARResNet44(**kwargs): return Img_CIFARResNet(7, tt=False, **kwargs)
def Img_CIFARResNet56(**kwargs): return Img_CIFARResNet(9, tt=False, **kwargs)
def Img_CIFARResNet110(**kwargs): return Img_CIFARResNet(18, tt=False, **kwargs)
def Img_CIFARResNet1202(**kwargs): return Img_CIFARResNet(200, tt=False, **kwargs)

def Img_CIFARResNet20_TT(**kwargs): return Img_CIFARResNet(3, tt=True, **kwargs)
def Img_CIFARResNet32_TT(**kwargs): return Img_CIFARResNet(5, tt=True, **kwargs)
def Img_CIFARResNet44_TT(**kwargs): return Img_CIFARResNet(7, tt=True, **kwargs)
def Img_CIFARResNet56_TT(**kwargs): return Img_CIFARResNet(9, tt=True, **kwargs)
def Img_CIFARResNet110_TT(**kwargs): return Img_CIFARResNet(18, tt=True, **kwargs)
def Img_CIFARResNet1202_TT(**kwargs): return Img_CIFARResNet(200, tt=True, **kwargs)


def Img_CIFARResNet20_TT_last(**kwargs):
    model = Img_CIFARResNet20()
    model.layer3[-1].conv2 = BasicBlock_TT._create_conv_layer(
        (4, 4, 4), (4, 4, 4), stride=1, **kwargs
    )
    return model
