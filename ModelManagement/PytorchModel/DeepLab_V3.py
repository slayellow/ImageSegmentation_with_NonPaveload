import UtilityManagement.config as cf
from ModelManagement.PytorchModel.ResNet import *
from ModelManagement.PytorchModel.ASPP import *
from ModelManagement.PytorchModel.Decoder import *
import math
import os
import warnings


class DeepLab_V3(nn.Module):
    def __init__(self, layer_num, classes):
        super(DeepLab_V3, self).__init__()

        self.model_name = 'DeepLab_V3'

        self.resnet = ResNet101(layer_num, 1000)
        self.aspp = ASPP()
        self.decoder = Decoder(classes)

    def forward(self, input):
        x, low_level_feat = self.resnet(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_name(self):
        return self.model_name


def DeepLab(layer_num, classes):
    pretrained_path = cf.paths['pretrained_path']
    model = DeepLab_V3(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        print('Pretrained Model!')
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])

    return model

#
# model = DeepLab(101, cf.NUM_CLASSES)
# model.eval()
# input = torch.rand([1, 3, 1052, 1914])
# output = model(input)
# print(output.size())