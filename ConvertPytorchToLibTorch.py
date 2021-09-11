
import os
import UtilityManagement.config as cf
from UtilityManagement.pytorch_util import *
from ModelManagement.PytorchModel.DeepLab_V3 import DeepLab

gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

model = DeepLab(101, cf.NUM_CLASSES).to(devices)

pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    load_weight_parameter(model, checkpoint['state_dict'])
else:
    print("No Pretrained Model")

# Don't forget change model to eval mode
model.eval()
example = torch.rand(1, 3, 480, 640).to(devices)

# trace weight
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/home/HONG/PretrainedParameter/DeepLab_V3_CPlusPlus.pt")
