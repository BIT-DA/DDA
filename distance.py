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


# sliced wasserstein computation use
def get_theta(embedding_dim, num_samples=50):
    theta = [
        w / np.sqrt((w**2).sum())
        for w in np.random.normal(size=(num_samples, embedding_dim))
    ]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor).cuda()

def sliced_wasserstein_distance(feature1,
                                feature2,
                                embed_dim=12,
                                num_projections=256,
                                p=1):
    theta = get_theta(embed_dim, num_projections)
    proj_target = feature2.matmul(theta.transpose(0, 1))
    proj_source = feature1.matmul(theta.transpose(0, 1))
    w_distance = torch.sort(proj_target.transpose(
        0, 1), dim=1)[0] - torch.sort(proj_source.transpose(0, 1), dim=1)[0]

    # calculate by the definition of p-Wasserstein distance
    w_distance_p = torch.pow(w_distance, p)
    print(w_distance_p.mean().size())
    return w_distance_p.sum(dim=1)


def MeanWssersteinDistance(feature):
    cls_size = len(feature)
    mean = feature[0].clone()
    for i in range(cls_size):
        if i != 0:
            mean += feature[i]
    distance = sliced_wasserstein_distance(mean, feature[0])
    for i in range(cls_size):
        if i != 0:
            distance += sliced_wasserstein_distance(mean, feature[i])

    return distance


def ListWssersteinDistance(feature):
    cls_size = len(feature)
    mean = feature[0].clone()
    for i in range(cls_size):
        if i != 0:
            mean += feature[i]
    distance = 0.0
    for i in range(cls_size):
        if i != 0:
            distance += sliced_wasserstein_distance(feature[i - 1], feature[i])

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
    "Mw": MeanWssersteinDistance,
    "Lw": ListWssersteinDistance,
    "kl": MeanKLDistance,
    "mse": MeanMSEDistance,
    "1prob": FirstProb,
    "-1prob": LastProb
}
