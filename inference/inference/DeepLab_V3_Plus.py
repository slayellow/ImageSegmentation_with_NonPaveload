import inference.config as cf
from inference.Xecption import *
from inference.ASPP import *
from inference.Decoder import *
import math
import os
import warnings


class DeepLab_V3_Plus(nn.Module):
    def __init__(self, classes):
        super(DeepLab_V3_Plus, self).__init__()

        self.model_name = 'DeepLab_V3_Plus'

        self.xecption = load_Xception(1000)
        self.aspp = ASPP()
        self.decoder = Decoder(classes, "xecption")

    def forward(self, input):
        x, low_level_feat = self.xecption(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_name(self):
        return self.model_name


def DeepLabV3Plus(classes):
    pretrained_path = cf.paths['pretrained_path']
    model = DeepLab_V3_Plus(classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        print('Pretrained Model!')
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])

    return model
