import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import math


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find(
            'ConvTranspose2d') != -1:
        # nn.init.kaiming_uniform_(m.weight)
        # nn.init.zeros_(m.bias)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LRN(nn.Module):
    def __init__(self,
                 local_size=1,
                 alpha=1.0,
                 beta=0.75,
                 ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int(
                                            (local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


resnet_dict = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
    "ResNet152": models.resnet152
}


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class ResNetFc(nn.Module):
    def __init__(self,
                 resnet_name,
                 use_bottleneck=True,
                 bottleneck_dim=256,
                 new_cls=False,
                 class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features,
                                            bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{
                    "params": self.feature_layers.parameters(),
                    "lr_mult": 1,
                    'decay_mult': 2
                }, {
                    "params": self.bottleneck.parameters(),
                    "lr_mult": 10,
                    'decay_mult': 2
                }, {
                    "params": self.fc.parameters(),
                    "lr_mult": 10,
                    'decay_mult': 2
                }]
            else:
                parameter_list = [{
                    "params": self.feature_layers.parameters(),
                    "lr_mult": 1,
                    'decay_mult': 2
                }, {
                    "params": self.fc.parameters(),
                    "lr_mult": 10,
                    'decay_mult': 2
                }]
        else:
            parameter_list = [{
                "params": self.parameters(),
                "lr_mult": 1,
                'decay_mult': 2
            }]
        return parameter_list


vgg_dict = {
    "VGG11": models.vgg11,
    "VGG13": models.vgg13,
    "VGG16": models.vgg16,
    "VGG19": models.vgg19,
    "VGG11BN": models.vgg11_bn,
    "VGG13BN": models.vgg13_bn,
    "VGG16BN": models.vgg16_bn,
    "VGG19BN": models.vgg19_bn
}


class VGGFc(nn.Module):
    def __init__(self,
                 vgg_name,
                 use_bottleneck=True,
                 bottleneck_dim=256,
                 new_cls=False,
                 class_num=1000):
        super(VGGFc, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(4096, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(4096, class_num)
                self.fc.apply(init_weights)
                self.__in_features = 4096
        else:
            self.fc = model_vgg.classifier[6]
            self.__in_features = 4096

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{
                    "params": self.features.parameters(),
                    "lr_mult": 1,
                    'decay_mult': 2
                }, {
                    "params": self.classifier.parameters(),
                    "lr_mult": 1,
                    'decay_mult': 2
                }, {
                    "params": self.bottleneck.parameters(),
                    "lr_mult": 10,
                    'decay_mult': 2
                }, {
                    "params": self.fc.parameters(),
                    "lr_mult": 10,
                    'decay_mult': 2
                }]
            else:
                parameter_list = [{
                    "params": self.feature_layers.parameters(),
                    "lr_mult": 1,
                    'decay_mult': 2
                }, {
                    "params": self.classifier.parameters(),
                    "lr_mult": 1,
                    'decay_mult': 2
                }, {
                    "params": self.fc.parameters(),
                    "lr_mult": 10,
                    'decay_mult': 2
                }]
        else:
            parameter_list = [{
                "params": self.parameters(),
                "lr_mult": 1,
                'decay_mult': 2
            }]
        return parameter_list


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024, nBlocks=5):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.blocks_num = nBlocks
        self.output_dim = output_dim
        # using the same random matrix at all exits
        self.random_matrix = [
            torch.randn(input_dim_list[i], output_dim)
            for i in range(self.input_num)
        ]

    def forward(self, input_list):
        tensor_list = []
        for j in range(self.blocks_num):
            return_list = [
                torch.mm(input_list[i][j], self.random_matrix[i])
                for i in range(self.input_num)
            ]
            return_tensor = return_list[0] / math.pow(float(self.output_dim),
                                                      1.0 / len(return_list))
            for single in return_list[1:]:
                return_tensor = torch.mul(return_tensor, single)
            tensor_list.append(return_tensor)
        return tensor_list

    def cuda(self):
        super(RandomLayer, self).cuda()
        for i in range(self.input_num):
            self.random_matrix[i] = self.random_matrix[i].cuda()


class GroupAdversarialNetworks(nn.Module):
    def __init__(
        self,
        nblocks=5,
        channel=[],
    ):
        super(GroupAdversarialNetworks, self).__init__()
        if not channel:
            channel = []
        self.blocks = nblocks
        self.Adversarials = nn.ModuleList()
        for i in range(nblocks):
            self.Adversarials.append(
                self._build_adversarial_network(channel[i], 1024))

    def _build_adversarial_network(slef, nIn, hidden_size):
        return AdversarialNetwork(in_feature=nIn, hidden_size=hidden_size)

    def forward(self, x):
        domain = []
        for i in range(self.blocks):
            domain.append(self.Adversarials[i](x[i]))
        return domain

    def get_parameters(self):
        parameter_list = []
        for i in range(self.blocks):
            parameter_list.append(self.Adversarials[i].get_parameters())
        return parameter_list


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha,
                           self.max_iter)
        # x = x.float()
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return {"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}


class ReverseLayerF(torch.autograd.Function):
    """Gradient Reverse Layer(Unsupervised Domain Adaptation by Backpropagation)
    Definition: During the forward propagation, GRL acts as an identity transform. During the back propagation though,
    GRL takes the gradient from the subsequent level, multiplies it by -alpha  and pass it to the preceding layer.

    Args:
        x (Tensor): the input tensor
        alpha (float): \alpha =  \frac{2}{1+\exp^{-\gamma \cdot p}}-1 (\gamma =10)
        out (Tensor): the same output tensor as x
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GroupClassifiers(nn.Module):
    def __init__(
        self,
        nblocks=5,
        num_classes=345,
        num_layer=1,
        channel=[],
    ):
        super(GroupClassifiers, self).__init__()
        if not channel:
            channel = []
        self.blocks = nblocks
        self.Classifiers = nn.ModuleList()
        self.num_layer = num_layer
        for i in range(self.blocks):
            self.Classifiers.append(
                self._build_classifier(channel[i], num_classes, 1024))

    def _build_classifier(self, nIn, classes, hidden_size):
        return Classifier(num_classes=classes,
                          num_unit=nIn,
                          middle=hidden_size,
                          num_layer=self.num_layer)

    def forward(self, x):
        pred = []
        for i in range(self.blocks):
            pred.append(self.Classifiers[i](x[i]))
        return pred

    def get_parameters(self):
        parameter_list = []
        for i in range(self.blocks):
            parameter_list.append(self.Classifiers[i].get_parameters())
        return parameter_list


class Classifier(nn.Module):
    def __init__(self,
                 num_classes=345,
                 prob=0.5,
                 num_layer=1,
                 num_unit=256,
                 middle=1024):
        super(Classifier, self).__init__()
        layers = []
        if num_layer == 1:
            layers.append(nn.Linear(num_unit, num_classes))
        else:
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer - 1):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle, middle))
                layers.append(nn.BatchNorm1d(middle, affine=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(middle, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.classifier.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        y = self.classifier(x)
        return y

    def get_parameters(self):
        return {"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}
