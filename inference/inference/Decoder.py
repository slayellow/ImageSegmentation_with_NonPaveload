from inference.pytorch_util import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone='xecption'):
        super(Decoder, self).__init__()

        if backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'resnet':
            low_level_inplanes = 256
        else:
            low_level_inplanes = 128

        self.conv1 = set_conv(low_level_inplanes, 48, kernel=1, strides=1, padding=0, bias=False)
        self.bn1 = set_batch_normalization(48)
        self.relu = set_relu()
        self.last_conv = nn.Sequential(set_conv(304, 256, kernel=3, strides=1, padding=1,bias=False),
                                       set_batch_normalization(256),
                                       set_relu(),
                                       set_dropout(0.5),
                                       set_conv(256, 256, kernel=3, strides=1, padding=1, bias=False),
                                       set_batch_normalization(256),
                                       set_relu(),
                                       set_dropout(0.1),
                                       set_conv(256, num_classes, kernel=1, strides=1, padding=0))

        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = set_concat((x, low_level_feat), axis=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
