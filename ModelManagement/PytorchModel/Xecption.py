from UtilityManagement.pytorch_util import *
import UtilityManagement.config as cf
import os.path
import math
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = set_detphwise_conv(in_channels, in_channels, kernel=kernel_size, strides=stride, padding=1,
                                        dilation=dilation, bias=bias)
        self.bn = set_batch_normalization(in_channels)
        self.pointwise = set_pointwise_conv(in_channels, out_channels, kernel=1, strides=1, padding=0, dilation=1,
                                            bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = set_conv(in_filters, out_filters, kernel=1, strides=strides, padding=0, dilation=1, bias=False)
            self.skipbn = set_batch_normalization(out_filters)
        else:
            self.skip = None

        self.relu = set_relu(use_input=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, dilation))
            rep.append(set_batch_normalization(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
            rep.append(set_batch_normalization(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, dilation))
            rep.append(set_batch_normalization(out_filters))

        if strides != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_filters, out_filters, 3, 2))
            rep.append(set_batch_normalization(out_filters))

        if strides == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_filters, out_filters, 3, 1))
            rep.append(set_batch_normalization(out_filters))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        entry_block3_stride = 2
        middle_block_dilation = 1
        exit_block_dilations = (1, 2)

        self.num_classes = num_classes
        self.model_name = 'Xception'

        self.conv1 = set_conv(3, 32, kernel=3, strides=2, padding=1, bias=False)
        self.bn1 = set_batch_normalization(32)
        self.relu = set_relu(use_input=True)

        self.conv2 = set_conv(32, 64, kernel=3, strides=1, padding=1, bias=False)
        self.bn2 = set_batch_normalization(64)
        # do relu here

        self.block1 = Block(64, 128, reps=2, strides=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, strides=2, start_with_relu=False)
        self.block3 = Block(256, 728, reps=2, strides=entry_block3_stride, is_last=True)

        # Middle flow
        self.block4 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block5 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block6 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block7 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block8 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block9 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block10 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block11 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block12 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block13 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block14 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block15 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block16 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block17 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block18 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)
        self.block19 = Block(728, 728, reps=3, strides=1, dilation=middle_block_dilation)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, strides=1, dilation=exit_block_dilations[0],
                                start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = set_batch_normalization(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = set_batch_normalization(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = set_batch_normalization(2048)

        self.fc = set_dense(2048, num_classes)


    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x, low_level_feat

    def get_name(self):
        return self.model_name

    def initialize_weights(self, init_weights):
        if init_weights is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                    if m.bias is not None:
                        m.bias.data.zero_()

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


def load_Xception(classes):
    pretrained_path = cf.paths['pretrained_path']
    model = Xception(classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model

