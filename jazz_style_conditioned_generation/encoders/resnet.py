#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet performer identification model, from https://github.com/HuwCheston/deep-pianist-identification"""

import os

import torch
import torch.nn as nn

from jazz_style_conditioned_generation import utils

# This maps classes used to train the performer identification model onto unique indices
_CLASS_MAPPING = {
    "Abdullah Ibrahim": 0,
    "Ahmad Jamal": 1,
    "Bill Evans": 2,
    "Brad Mehldau": 3,
    "Cedar Walton": 4,
    "Chick Corea": 5,
    "Gene Harris": 6,  # nb. no performer token, not used to condition generative model
    "Geri Allen": 7,  # nb. no performer token
    "Hank Jones": 8,
    "John Hicks": 9,
    "Junior Mance": 10,
    "Keith Jarrett": 11,
    "Kenny Barron": 12,
    "Kenny Drew": 13,
    "McCoy Tyner": 14,
    "Oscar Peterson": 15,
    "Stanley Cowell": 16,  # nb. no performer token
    "Teddy Wilson": 17,
    "Thelonious Monk": 18,
    "Tommy Flanagan": 19,
}


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(inplanes, width)

        # With ResNet50-IBN, only replace the first norm in each block with IBN
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, layers=None, ):
        super(ResNet50, self).__init__()
        # Use ResNet50 by default
        if layers is None:
            layers = [3, 4, 6, 3]

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2, dilate=False)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.tensor) -> torch.tensor:
        """Returns feature embeddings prior to final linear projection layer(s)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.forward_features(x)
        x = self.fc(x)
        return x


def load_performer_identifier(
        path: str = os.path.join(utils.get_project_root(), "references/resnet50-performer-identifier.pth")
) -> ResNet50:
    """Loads the pretrained performer identification model + pretrained weights"""
    # Initialise the model: this model was trained with 20 classes initially
    ident = ResNet50(num_classes=len(list(_CLASS_MAPPING.keys()))).to(utils.DEVICE)
    # Load the checkpoint and model state dictionary
    chkpt = torch.load(path, map_location=utils.DEVICE, weights_only=False)
    ident.load_state_dict(chkpt["model_state_dict"])
    # Freeze the model weights
    ident.eval()
    for param in ident.parameters():
        param.requires_grad = False
    return ident
