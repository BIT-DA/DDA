import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def Entropy(input_):
    # bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


def discrepancy(out1, out2):
    return torch.mean(
        torch.abs(nn.Softmax(dim=1)(out1) - nn.Softmax(dim=1)(out2)))

def Weighted_loss(logits, labels, weights):
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    target_flat = labels.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(labels.size()) * weights
    loss = losses.sum() / len(labels)
    return loss
