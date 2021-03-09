import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def MeanCosDistance(feature):
    cls_size = len(feature)
    mean = feature[0].clone()
    for i in range(cls_size):
        if i != 0:
            mean += feature[i]

    distance = torch.cosine_similarity(mean, feature[0], dim=1)
    for i in range(cls_size):
        if i != 0:
            distance += torch.cosine_similarity(mean, feature[i], dim=1)

    return distance


def klDivergence(feature1, feature2):
    p = F.softmax(feature1, dim=-1)
    _kl = torch.sum(
        p *
        (F.log_softmax(feature1, dim=-1) - F.log_softmax(feature2, dim=-1)), 1)
    return torch.mean(_kl)


def MeanKLDistance(feature):
    cls_size = len(feature)
    mean = feature[0].clone()
    for i in range(cls_size):
        if i != 0:
            mean += feature[i]

    distance = klDivergence(feature[0], mean)
    for i in range(cls_size):
        if i != 0:
            distance += klDivergence(feature[i], mean)

    return distance


def MSE(feature1, feature2):
    _mse = nn.MSELoss(reduction='none')(feature1, feature2)
    return torch.sum(_mse)


def MeanMSEDistance(feature):
    cls_size = len(feature)
    mean = feature[0].clone()
    for i in range(cls_size):
        if i != 0:
            mean += feature[i]

    distance = MSE(feature[0], mean)
    for i in range(cls_size):
        if i != 0:
            distance += MSE(feature[i], mean)

    return distance


def FirstProb(feature):
    confidence, predict = torch.max(feature[0], 1)
    return confidence


def LastProb(feature):
    confidence, predict = torch.max(feature[-1], 1)
    return confidence


distance_dict = {
    "cos": MeanCosDistance,
    "kl": MeanKLDistance,
    "mse": MeanMSEDistance,
    "1prob": FirstProb,
    "-1prob": LastProb
}
