import math
import torch
import torch.nn as nn


# ---- GradientRescale ---- #
class GradientRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output

        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply

# ---- END Gradient Rescale ---- #


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn,
                      nOut,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding,
                      bias=False), nn.BatchNorm2d(nOut), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck, bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(
                nn.Conv2d(nIn,
                          nInner,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(
                nn.Conv2d(nInner,
                          nOut,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
        elif type == 'down':
            layer.append(
                nn.Conv2d(nInner,
                          nOut,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down', bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal', bottleneck,
                                  bnWidth2)

    def forward(self, x):
        res = [x[1], self.conv_down(x[0]), self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal', bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        conv = nn.Sequential(
            nn.Conv2d(nIn, nOut * args["params"]["grFactor"][0], 7, 2, 3),
            nn.BatchNorm2d(nOut * args["params"]["grFactor"][0]),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layers.append(conv)

        nIn = nOut * args["params"]["grFactor"][0] 

        for i in range(1, args["params"]["nScales"]):
            self.layers.append(
                ConvBasic(nIn,
                          nOut * args["params"]["grFactor"][i],
                          kernel=3,
                          stride=2,
                          padding=1))
            nIn = nOut * args["params"]["grFactor"][i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args["params"][
            "nScales"]
        self.outScales = outScales if outScales is not None else args[
            "params"]["nScales"]

        self.nScales = args["params"]["nScales"]
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args["params"]["grFactor"][self.offset - 1]
            nIn2 = nIn * args["params"]["grFactor"][self.offset]
            _nOut = nOut * args["params"]["grFactor"][self.offset]
            self.layers.append(
                ConvDownNormal(nIn1, nIn2, _nOut, args["params"]["bottleneck"],
                               args["params"]["bnFactor"][self.offset - 1],
                               args["params"]["bnFactor"][self.offset]))
        else:
            self.layers.append(
                ConvNormal(nIn * args["params"]["grFactor"][self.offset],
                           nOut * args["params"]["grFactor"][self.offset],
                           args["params"]["bottleneck"],
                           args["params"]["bnFactor"][self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args["params"]["grFactor"][i - 1]
            nIn2 = nIn * args["params"]["grFactor"][i]
            _nOut = nOut * args["params"]["grFactor"][i]
            self.layers.append(
                ConvDownNormal(nIn1, nIn2, _nOut, args["params"]["bottleneck"],
                               args["params"]["bnFactor"][i - 1],
                               args["params"]["bnFactor"][i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierCustom(nn.Module):
    def __init__(self, m, nIn, use_bottleneck=False, bottleneck_dim=256):
        super(ClassifierCustom, self).__init__()
        self.m = m
        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = nn.Linear(nIn, bottleneck_dim)
        else:
            self.linear = nn.Linear(nIn, 1)

    def forward(self, x):
        x = self.m(x[-1])
        x = x.view(x.size(0), -1)
        if self.use_bottleneck:
            res = self.bottleneck(x)
            return res
        else:
            res = x
            x = self.linear(x) #used for budgeted computation
            return res


class MSDNet(nn.Module):
    def __init__(self, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.nBlocks = args["params"]["nBlocks"]
        self.steps = [args["params"]["base"]]
        self.__in_features = []
        self.args = args

        n_layers_all, n_layer_curr = args["params"]["base"], 0
        for i in range(1, self.nBlocks):
            self.steps.append(
                args["params"]["step"] if args["params"]["stepmode"] ==
                'even' else args["params"]["step"] * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args["params"]["nChannels"]
        use_bottleneck = args["params"]["use_bottleneck"]
        if use_bottleneck:
            bottleneck_dim = args["params"]["bottleneck_dim"]
        else:
            bottleneck_dim= 0
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]
            self.classifiers.append(
                self._build_custom_classifier(
                    nIn * args["params"]["grFactor"][-1], use_bottleneck,
                    bottleneck_dim))
            if not use_bottleneck:
                self.__in_features.append(nIn * args["params"]["grFactor"][-1])
            else:
                self.__in_features.append(bottleneck_dim)

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args["params"]["nScales"]
            outScales = args["params"]["nScales"]
            if args["params"]["prune"] == 'min': 
                inScales = min(args["params"]["nScales"],
                               n_layer_all - n_layer_curr + 2)
                outScales = min(args["params"]["nScales"],
                                n_layer_all - n_layer_curr + 1)
            elif args["params"]["prune"] == 'max':
                interval = math.ceil(1.0 * n_layer_all /
                                     args["params"]["nScales"])
                inScales = args["params"]["nScales"] - math.floor(
                    1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args["params"]["nScales"] - math.floor(
                    1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(
                MSDNLayer(nIn, args["params"]["growthRate"], args, inScales,
                          outScales))
            print(
                '|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'
                .format(inScales, outScales, nIn,
                        args["params"]["growthRate"]))

            nIn += args["params"]["growthRate"]
            if args["params"]["prune"] == 'max' and inScales > outScales and \
                    args["params"]["reduction"] > 0:
                offset = args["params"]["nScales"] - outScales
                layers.append(
                    self._build_transition(
                        nIn,
                        math.floor(1.0 * args["params"]["reduction"] * nIn),
                        outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args["params"]["reduction"] * nIn)
                print(
                    '|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'
                    .format(_t,
                            math.floor(1.0 * args["params"]["reduction"] *
                                       _t)))
            elif args["params"]["prune"] == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args["params"]["nScales"] - outScales
                layers.append(
                    self._build_transition(
                        nIn,
                        math.floor(1.0 * args["params"]["reduction"] * nIn),
                        outScales, offset, args))

                nIn = math.floor(1.0 * args["params"]["reduction"] * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(
                ConvBasic(nIn * args["params"]["grFactor"][offset + i],
                          nOut * args["params"]["grFactor"][offset + i],
                          kernel=1,
                          stride=1,
                          padding=0))
        return ParallelModule(net)

    def _build_custom_classifier(self, nIn, use_bottleneck, bottleneck_dim):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2))
        return ClassifierCustom(conv, nIn, use_bottleneck, bottleneck_dim)

    def forward(self, x):
        feature = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)

            x[-1] = gradient_rescale(x[-1], 1.0 / (self.nBlocks - i))
            feature_i = self.classifiers[i](x)
            x[-1] = gradient_rescale(x[-1], (self.nBlocks - i - 1))

            feature.append(feature_i)
        return feature

    def get_parameters(self):
        parameter_list = []
        for i in range(self.nBlocks):
            parameter_list.append({
                "params": self.blocks[i].parameters(),
                "lr_mult": 0.1,
                'decay_mult': 2
            })
            parameter_list.append({
                "params": self.classifiers[i].parameters(),
                "lr_mult": 1,
                'decay_mult': 2
            })
        return parameter_list

    def output_num(self):
        return self.__in_features

    def get_nBlocks(self):
        return self.nBlocks
