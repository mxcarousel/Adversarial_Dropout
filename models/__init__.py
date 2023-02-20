from .resnet import ResNet18
from .vgg import vgg11_bn
from .logstic import Logistic
from .mlp import mlp
from .lenet import LeNet

def get_model(model_name, input_size, classes):
    if model_name == "ResNet18":
        return ResNet18(input_size[0], classes)
    elif model_name == "VGG11BN":
        return vgg11_bn(input_size[0], classes)
    elif model_name == "Logistic":
        size = input_size[1]*input_size[2]#*input_size[2]
        return Logistic(size,classes)
    elif model_name == "mlp":
        size = input_size[0]*input_size[1]*input_size[2]
        return mlp(size, 1000, classes)
    elif model_name == "lenet":
        return LeNet(input_size, classes)
