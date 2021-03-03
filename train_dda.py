import argparse
import os
import os.path as osp
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lr_schedule
import distance
import pre_process as prep
from torch.utils.data import DataLoader
from data_list import ImageList
from data_list import MinHeap
import msdnet
import network
import loss
import random
from op_counter import measure_model
from adaptive_inference import dynamic_evaluate
visda_classes = [
    'aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
    'person', 'plant', 'skateboard', 'train', 'truck'
]


def image_classification_test(loader, model, classifier, test_10crop=True):
    model.train(False)
    classifier.train(False)
    start_test = True
    with torch.no_grad():
        if test_10crop:
            all_output = [[] for i in range(model.get_nBlocks())]
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = [[] for i in range(model.get_nBlocks())]
                for j in range(10):
                    feature = model(inputs[j])
                    predict_out = classifier(feature)
                    for k in range(model.get_nBlocks()):
                        outputs[k].append(nn.Softmax(dim=1)(predict_out[k]))
                for k in range(model.get_nBlocks()):
                    outputs[k] = sum(outputs[k])
                if start_test:
                    for k in range(model.get_nBlocks()):
                        all_output[k] = outputs[k].float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    for k in range(model.get_nBlocks()):
                        all_output[k] = torch.cat(
                            (all_output[k], outputs[k].float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            all_output = []
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                features = model(inputs)
                outputs = classifier(features)
                if start_test:
                    for j in range(model.get_nBlocks()):
                        outputs_ever = nn.Softmax(dim=1)(
                            outputs[j]).float().cpu()
                        all_output.append(outputs_ever)
                    all_label = labels.float()
                    start_test = False
                else:
                    for j in range(model.get_nBlocks()):
                        all_output[j] = torch.cat(
                            (all_output[j], nn.Softmax(dim=1)(
                                outputs[j]).float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    pre = []
    accuracy = []
    if config['dataset'] == 'visda':
        accuracy = [{} for i in range(model.get_nBlocks())]
        for i in range(model.get_nBlocks()):
            for j in visda_classes:
                accuracy[i][j] = []
                accuracy[i][j].append(0)
                accuracy[i][j].append(0)
            _, predict = torch.max(all_output[i], 1)
            for j in range(len(all_label)):
                key_label = visda_classes[all_label[j].long()]
                key_pred = visda_classes[predict[j].long()]
                accuracy[i][key_label][1] += 1
                if key_pred == key_label:
                    accuracy[i][key_pred][0] += 1
    else:
        for i in range(model.get_nBlocks()):
            _, predict = torch.max(all_output[i], 1)
            pre_num = predict.cpu().numpy()
            pre.append(pre_num)
            accuracy.append(
                torch.sum(torch.squeeze(predict).float() == all_label).item() /
                float(all_label.size()[0]))

    return accuracy


# generate pseudo label
def selfDetection(loader, model, classifier, config, iteration):
    model.train(False)
    classifier.train(False)
    distance_heap = MinHeap(key=lambda item: item[2])
    iter_test = iter(loader["detection"])
    cal_distance = distance.distance_dict[config["self-training"]["distance"]]
    class_num = config["network"]["params"]["class_num"]
    sum_classdistance = [0] * (class_num * 2)
    sum_class = [0] * (class_num * 2)
    with torch.no_grad():
        for i in range(len(loader['detection'])):
            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()
            label = data[1]
            path = data[2]
            features = model(inputs)
            outputs = classifier(features)
            # softmax
            pre = []
            predict_out = nn.Softmax(dim=1)(outputs[0])
            mean = predict_out.clone()
            pre.append(predict_out.clone())
            for j in range(model.get_nBlocks()):
                if j != 0:
                    predict_out = nn.Softmax(dim=1)(outputs[j])
                    pre.append(predict_out.clone())
                    mean += predict_out.clone()

            _distance = cal_distance(pre)
            mean = mean / model.get_nBlocks()
            confidence, predict = torch.max(mean, 1)
            if config['self-training']["is_confidence"]:
                _distance = _distance * confidence
            if len(path) == 1:
                distance_heap.push([
                    path[0],
                    torch.squeeze(predict).cpu().numpy(),
                    -_distance.cpu().numpy(), label
                ])
            else:
                for j in range(len(path)):
                    pre_label = torch.squeeze(predict).cpu().numpy()[j]
                    sum_class[pre_label] += 1
                    sum_classdistance[pre_label] += _distance.cpu().numpy()[j]
                    distance_heap.push([
                        path[j], pre_label, -_distance.cpu().numpy()[j],
                        label[j]
                    ])

        st_config = config["self-training"]
        if st_config["scale"] <= st_config["maxscale"]:
            st_config["scale"] = st_config["scale"] + st_config["increase"]
        to_source_size = len(distance_heap) * st_config["scale"]
        size_class = [0] * (class_num * 2)
        if st_config["class_balance"] == "distance_ratio":
            sum_mean = 0.0
            for i in range(class_num):
                if sum_class[i] != 0:
                    size_class[i] = sum_classdistance[i] / sum_class[i]
                    sum_mean += size_class[i]
            for i in range(class_num):
                size_class[i] = int(to_source_size *
                                    (size_class[i] / sum_mean))
        else:
            size_class = [to_source_size / class_num] * (class_num * 2)
        sum_class = [0] * (class_num * 2)
        pseudo_path = config["output_path"] + "/pseudo" + str(
            iteration) + ".txt"
        pseudo = open(pseudo_path, "w")
        correct = 0
        pseudo_size = 0
        for i in range(len(distance_heap)):
            data = distance_heap.pop()
            if sum_class[int(data[1])] <= size_class[
                    data[1]] and data[2] <= st_config["threshold"]:
                sum_class[data[1]] += 1
                pseudo.flush()
                pseudo.write(data[0] + " ")
                pseudo.write("{:d} ".format(data[1]))
                pseudo_size += 1
                rand_cls = random.randint(0, model.get_nBlocks() - 1)
                if data[1] == data[3]:
                    correct += 1
                    pseudo.write("{:d} 1 {:.5f}\n".format(rand_cls, data[2]))
                else:
                    pseudo.write("{:d} 0 {:.5f}\n".format(rand_cls, data[2]))
        pseudo.close()
    return pseudo_path, pseudo_size, correct


def train(config):
    # set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    # prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"] * 2
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(
        data_config["source"]["list_path"]).readlines(),
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"],
                                        batch_size=int(train_bs / 2),
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True)
    dsets["target"] = ImageList(open(
        data_config["target"]["list_path"]).readlines(),
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=int(train_bs / 2),
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [
                ImageList(open(data_config["test"]["list_path"]).readlines(),
                          transform=prep_dict["test"][i]) for i in range(10)
            ]
            dset_loaders["test"] = [
                DataLoader(dset,
                           batch_size=test_bs,
                           shuffle=False,
                           num_workers=4) for dset in dsets['test']
            ]
            dsets["validation"] = [
                ImageList(open(data_config["target"]["list_path"]).readlines(),
                          transform=prep_dict["test"][i]) for i in range(10)
            ]
            dset_loaders["validation"] = [
                DataLoader(dset,
                           batch_size=test_bs,
                           shuffle=False,
                           num_workers=4) for dset in dsets['validation']
            ]
    else:
        dsets["test"] = ImageList(open(
            data_config["test"]["list_path"]).readlines(),
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"],
                                          batch_size=test_bs,
                                          shuffle=False,
                                          num_workers=4)
        dsets["validation"] = ImageList(open(
            data_config["target"]["list_path"]).readlines(),
                                        transform=prep_dict["test"])
        dset_loaders["validation"] = DataLoader(dsets["validation"],
                                                batch_size=test_bs,
                                                shuffle=False,
                                                num_workers=4)

    prep_dict["detection"] = prep.image_test(**config["prep"]['params'])
    dsets["detection"] = ImageList(open(
        data_config["target"]["list_path"]).readlines(),
                                   transform=prep_dict["detection"])
    dset_loaders["detection"] = DataLoader(dsets["detection"],
                                           batch_size=test_bs,
                                           shuffle=False,
                                           num_workers=0)

    # set base network
    net_config = config["network"]

    base_network = msdnet.MSDNet(net_config)
    if net_config["pattern"] == "budget":
        IM_SIZE = 224
        n_flops, n_params = measure_model(base_network, IM_SIZE, IM_SIZE)
        torch.save(n_flops, os.path.join(config["output_path"], 'flops.pth'))
    base_network = base_network.cuda()
    state_dict = torch.load(net_config["preTrained"])['state_dict']
    state_dict_adapt = {}
    for key in state_dict.keys():
        if key[:17] == "module.classifier":
            pass
        else:
            state_dict_adapt[key[7:]] = state_dict[key]

    # set base_network
    base_network.load_state_dict(state_dict_adapt, strict=False)
    # set classifier
    classifier = network.GroupClassifiers(
        nblocks=base_network.get_nBlocks(),
        num_classes=config["network"]["params"]["class_num"],
        channel=base_network.output_num())
    classifier = classifier.cuda()

    # set adversrial
    ad_net = network.GroupAdversarialNetworks(
        nblocks=base_network.get_nBlocks(), channel=base_network.output_num())
    ad_net = ad_net.cuda()

    parameter_list = base_network.get_parameters() + classifier.get_parameters(
    ) + ad_net.get_parameters()

    # set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list,
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # set self-training
    st_select = 0
    st_config = config["self-training"]
    has_pseudo = False

    # train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    source_iter = 0
    pseudo_iter = 0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            temp_acc = image_classification_test(
                dset_loaders,
                base_network,
                classifier,
                test_10crop=prep_config["test_10crop"])
            if config['dataset'] == 'visda':
                log_str = "iter: {:05d}".format(i)
                correct = [0.0 for i in range(base_network.get_nBlocks())]
                allsample = [0.0 for i in range(base_network.get_nBlocks())]
                meanclass = [0.0 for i in range(base_network.get_nBlocks())]
                for j in range(base_network.get_nBlocks()):
                    log_str += "\n {} classifier:".format(j)
                    for k in visda_classes:
                        correct[j] += temp_acc[j][k][0]
                        allsample[j] += temp_acc[j][k][1]
                        meanclass[
                            j] += 100. * temp_acc[j][k][0] / temp_acc[j][k][1]
                        log_str += '\t{}: [{}/{}] ({:.6f}%)'.format(
                            k, temp_acc[j][k][0], temp_acc[j][k][1],
                            100. * temp_acc[j][k][0] / temp_acc[j][k][1])
                log_str += "\nall: "
                for j in range(base_network.get_nBlocks()):
                    log_str += "{:02d}-pre:{:.05f}".format(
                        j, 100. * correct[j] / allsample[j])
                log_str += "\ncls: "
                for j in range(base_network.get_nBlocks()):
                    log_str += "{:02d}-pre:{:.05f}".format(
                        j, meanclass[j] /
                        config["network"]["params"]["class_num"])
                config["out_file"].write(log_str + "\n")
                config["out_file"].flush()
            else:
                log_str = "iter: {:05d}".format(i)
                for j in range(base_network.get_nBlocks()):
                    log_str += " {:02d}-pre:{:.05f}".format(j, temp_acc[j])
                config["out_file"].write(log_str + "\n")
            print(log_str)

        if (i + 1) % config["snapshot_interval"] == 0 and config["save_model"]:
            torch.save(
                base_network,
                osp.join(config["output_path"],
                         "iter_{:05d}_model.pth.tar".format(i)))

        # dynamic evaluation
        if (i + 1) % 3000 == 0 and net_config["pattern"] == "budget":
            torch.multiprocessing.set_sharing_strategy('file_system')
            dynamic_evaluate(base_network, classifier,
                             dset_loaders["test"], dset_loaders["validation"],
                             config, 'target-validation@' + str(i))
            torch.multiprocessing.set_sharing_strategy('file_descriptor')

        if (source_iter % len_train_source == 0) and i >= st_config["start"]:
            st_select += 1
            has_pseudo = True
            pseudo_path, len_train_pseudo, correct = selfDetection(
                dset_loaders, base_network, classifier, config, i)
            log_str = "size: {} correct:{}".format(len_train_pseudo, correct)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
            # set batch
            source_batchsize = int(
                int(train_bs / 2) * len_train_source /
                (len_train_source + len_train_pseudo))
            if source_batchsize == int(train_bs / 2):
                source_batchsize -= 1
            if source_batchsize < int(int(train_bs / 2) / 2):
                source_batchsize = int(int(train_bs / 2) / 2)
            pseudo_batchsize = int(train_bs / 2) - source_batchsize

            dsets["source"] = ImageList(open(
                data_config["source"]["list_path"]).readlines(),
                                        transform=prep_dict["source"])
            dset_loaders["source"] = DataLoader(dsets["source"],
                                                batch_size=source_batchsize,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True)
            dsets["target"] = ImageList(open(
                data_config["target"]["list_path"]).readlines(),
                                        transform=prep_dict["target"])
            dset_loaders["target"] = DataLoader(dsets["target"],
                                                batch_size=source_batchsize,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True)

            dsets["pseudo"] = ImageList(open(pseudo_path).readlines(),
                                        transform=prep_dict["target"])
            dset_loaders["pseudo"] = DataLoader(dsets["pseudo"],
                                                batch_size=pseudo_batchsize,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True)
            dsets["source_pseudo"] = ImageList(open(
                data_config["source"]["list_path"]).readlines(),
                                               transform=prep_dict["source"])
            dset_loaders["source_pseudo"] = DataLoader(
                dsets["source_pseudo"],
                batch_size=pseudo_batchsize,
                shuffle=True,
                num_workers=4,
                drop_last=True)

            len_train_source = len(dset_loaders["source"])
            len_train_target = len(dset_loaders["target"])
            len_train_pseudo = len(dset_loaders["pseudo"])
            len_train_source_pseudo = len(dset_loaders["source_pseudo"])

            source_iter = 0
            pseudo_iter = 0

            iter_source = iter(dset_loaders["source"])
            iter_target = iter(dset_loaders["target"])
            iter_pseudo = iter(dset_loaders["pseudo"])
            iter_source_pseudo = iter(dset_loaders["source_pseudo"])
            # set self-training oprimizer
            if st_config["is_lr"] and st_select == 0:
                param_lr = []
                optimizer = optimizer_config["type"](
                    parameter_list, **(st_config["optimizer"]["optim_params"]))
                for param_group in optimizer.param_groups:
                    param_lr.append(param_group["lr"])
                schedule_param = st_config["optimizer"]["lr_param"]
                lr_scheduler = lr_schedule.schedule_dict[st_config["optimizer"]
                                                         ["lr_type"]]
        loss_params = config["loss"]
        # train one iter
        base_network.train(True)
        classifier.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if has_pseudo:
            if source_iter % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if source_iter % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])
            if pseudo_iter % len_train_pseudo == 0:
                iter_pseudo = iter(dset_loaders["pseudo"])
            if pseudo_iter % len_train_source_pseudo == 0:
                iter_source_pseudo = iter(dset_loaders["source_pseudo"])

            source_iter += 1
            pseudo_iter += 1

            inputs_source, labels_source, _, _ = iter_source.next()
            inputs_target, _, _, _ = iter_target.next()

            inputs_pseudo, labels_pseudo, _, randcls = iter_pseudo.next()
            inputs_source_pseudo, _, _, _ = iter_source_pseudo.next()

            inputs_source = torch.cat((inputs_source, inputs_pseudo), dim=0)
            labels_source = torch.cat((labels_source, labels_pseudo), dim=0)

            inputs_target = torch.cat((inputs_target, inputs_source_pseudo),
                                      dim=0)
        else:
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])

            inputs_source, labels_source, _, _ = iter_source.next()
            inputs_target, _, _, _ = iter_target.next()

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), \
            inputs_target.cuda(), \
            labels_source.cuda(),

        features_source = base_network(inputs_source)
        outputs_source = classifier(features_source)
        domain_source = ad_net(features_source)
        features_target = base_network(inputs_target)
        domain_target = ad_net(features_target)

        classifier_loss = 0.0
        transfer_loss = 0.0
        if has_pseudo and st_config["is_weight"]:
            mean_pseudo = nn.Softmax(dim=1)(
                outputs_source[0][source_batchsize:].detach())
            for j in range(base_network.get_nBlocks()):
                if j != 0:
                    mean_pseudo += nn.Softmax(dim=1)(
                        outputs_source[j][source_batchsize:].detach())
            mean_pseudo = mean_pseudo / base_network.get_nBlocks()
        for j in range(base_network.get_nBlocks()):
            if has_pseudo and st_config["is_weight"]:
                source_mask = torch.FloatTensor([1.] * source_batchsize)
                pseudo_mask = [1 if k == j else 0 for k in randcls]
                pseudo_mask = torch.tensor(pseudo_mask).float()
                mask = torch.cat((source_mask, pseudo_mask), dim=0).cuda()
                classifier_loss += loss.Weighted_loss(outputs_source[j],
                                                      labels_source, mask)
            else:
                classifier_loss += nn.CrossEntropyLoss()(outputs_source[j],
                                                         labels_source)
            domain = torch.cat((domain_source[j], domain_target[j]), dim=0)
            batch_size = domain.size(0) // 2
            if has_pseudo:
                dc_target = torch.from_numpy(
                    np.array([[1]] * source_batchsize +
                             [[0]] * pseudo_batchsize +
                             [[0]] * source_batchsize +
                             [[1]] * pseudo_batchsize)).float().cuda()
            else:
                dc_target = torch.from_numpy(
                    np.array([[1]] * batch_size +
                             [[0]] * batch_size)).float().cuda()
            transfer_loss += nn.BCELoss()(domain, dc_target)

        total_loss = classifier_loss + loss_params["trade_off"] * transfer_loss
        total_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument(
        '--dset',
        type=str,
        default='visda',
        choices=['visda', 'domainnet'],
        help="The dataset or source dataset used")
    parser.add_argument(
        '--s_dset_path',
        type=str,
        default='/data1/TL/data/list/visda2017/synthetic_12.txt',
        help="The source dataset path list")
    parser.add_argument('--t_dset_path',
                        type=str,
                        default='/data1/TL/data/list/visda2017/real_12.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_dset_path',
                        type=str,
                        default='/data1/TL/data/list/visda2017/real_12.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval',
                        type=int,
                        default=500,
                        help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval',
                        type=int,
                        default=10000,
                        help="interval of two continuous output model")
    parser.add_argument('--save_model',
                        default=False,
                        action='store_true',
                        help="save model")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='san',
        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr',
                        type=float,
                        default=0.015,
                        help="learning rate")
    parser.add_argument('--pattern',
                        type=str,
                        default='anytime',
                        choices=["anytime", "budget"])
    parser.add_argument('--ststart',
                        type=int,
                        default=10000,
                        help="the iterations of self-training start")
    parser.add_argument('--stinterval',
                        type=int,
                        default=3000,
                        help="the iterations of self-training start")
    parser.add_argument('--use_stlr',
                        default=False,
                        action='store_true',
                        help="is use self-training learning rate")
    parser.add_argument('--stlr',
                        type=float,
                        default=0.015,
                        help="use self-training learning rate")
    parser.add_argument('--stscale',
                        type=float,
                        default=0.8,
                        help="self-training start scale target to source")
    parser.add_argument('--stmaxscale',
                        type=float,
                        default=1.0,
                        help="self-training biggest scale target to source")
    parser.add_argument('--stincrease',
                        type=float,
                        default=0,
                        help="scale increase per epoch")
    parser.add_argument('--stclass_balance',
                        type=str,
                        default='distance_ratio',
                        choices=["mean", "distance_ratio"])
    parser.add_argument('--stthreshold',
                        type=float,
                        default=1e9,
                        help="distance should greater than this num")
    parser.add_argument(
        '--stdistance',
        type=str,
        default='cos',
        choices=["cos", "Mw", "Lw", "kl", "mse", "1prob", "-1prob"])
    parser.add_argument('--not_stweight',
                        default=False,
                        action='store_true',
                        help="is use weight for cls")
    parser.add_argument('--not_stconfidence',
                        default=False,
                        action='store_true',
                        help="is use confidence in self-training ")
    parser.add_argument('--trade_off',
                        type=float,
                        default=1.0,
                        help="trade off in loss")
    parser.add_argument('--max_iter',
                        type=int,
                        default=300000,
                        help="models train with the max iterations")

    # model arch related
    model_names = ['msdnet']
    arch = parser.add_argument_group('arch', 'model architecture setting')

    arch.add_argument('--arch',
                      '-a',
                      metavar='ARCH',
                      default='msdnet',
                      type=str,
                      choices=model_names,
                      help='model architecture: ' + ' | '.join(model_names) +
                      ' (default: MSDNet)')
    arch.add_argument('--reduction',
                      default=0.5,
                      type=float,
                      metavar='C',
                      help='compression ratio of DenseNet'
                      ' (1 means dot\'t use compression) (default: 0.5)')
    arch.add_argument(
        '--pretrain_path',
        type=str,
        default='../msdnet-step=4-block=5.pth.tar',
        help="The pre-trained model path")
    # msdnet config
    arch.add_argument('--nBlocks', type=int, default=5)
    arch.add_argument('--nChannels', type=int, default=32)
    arch.add_argument('--base', type=int, default=4)
    arch.add_argument('--stepmode',
                      type=str,
                      choices=['even', 'lin_grow'],
                      default='even')
    arch.add_argument('--step', type=int, default=4)
    arch.add_argument('--growthRate', type=int, default=16)
    arch.add_argument('--grFactor', default='1-2-4-4', type=str)
    arch.add_argument('--prune', default='max', choices=['min', 'max'])
    arch.add_argument('--bnFactor', default='1-2-4-4')
    arch.add_argument('--bottleneck', default=True, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    nowTime = datetime.datetime.now().strftime('%m-%d,%H:%M:%S')
    config = {}
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.max_iter
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["save_model"] = args.save_model
    config["output_path"] = "snapshot/" + args.output_dir + nowTime
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {
        "test_10crop": False,
        'params': {
            "resize_size": 256,
            "crop_size": 224,
            'alexnet': False
        }
    }
    config["loss"] = {
        "trade_off": args.trade_off
    }
    config["self-training"] = {
        "start": args.ststart,
        "scale": args.stscale,
        "maxscale": args.stmaxscale,
        "interval": args.stinterval,
        "increase": args.stincrease,
        "threshold": args.stthreshold,
        "distance": args.stdistance,
        "class_balance": args.stclass_balance,
        "is_lr": args.use_stlr,
        "lr": args.stlr,
        "is_weight":  bool(1 - args.not_stweight),
        "is_confidence": bool(1 - args.not_stconfidence),
        "optimizer": {
            "type": optim.SGD,
            "optim_params": {
                'lr': args.stlr / 10.0,
                "momentum": 0.9,
                "weight_decay": 0.0003,
                "nesterov": False
            },
            "lr_type": "inv",
            "lr_param": {
                "lr": args.stlr,
                "gamma": 10,
                "power": 0.75
            }
        }
    }

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    config["network"] = {
        "name": args.arch,
        "preTrained": args.pretrain_path,
        "pattern": args.pattern,
        "params": {
            "reduction": args.reduction,
            "nBlocks": args.nBlocks,
            "nChannels": args.nChannels,
            "base": args.base,
            "stepmode": args.stepmode,
            "step": args.step,
            "growthRate": args.growthRate,
            "grFactor": args.grFactor,
            "prune": args.prune,
            "bnFactor": args.bnFactor,
            "bottleneck": args.bottleneck,
            "use_bottleneck": False,
            "nScales": len(args.grFactor)
        }
    }

    config["optimizer"] = {
        "type": optim.SGD,
        "optim_params": {
            'lr': args.lr / 10.0,
            "momentum": 0.9,
            "weight_decay": 0.0003,
            "nesterov": False
        },  
        "lr_type": "inv",
        "lr_param": {
            "lr": args.lr,
            "gamma": 10,
            "power": 0.75
        }
    }

    config["dataset"] = args.dset
    config["data"] = {
        "source": {
            "list_path": args.s_dset_path,
            "batch_size": 36
        },
        "target": {
            "list_path": args.t_dset_path,
            "batch_size": 36
        },
        "test": {
            "list_path": args.test_dset_path,
            "batch_size": 36
        }
    }

    if config["dataset"] == "visda":
        # config["optimizer"]["lr_param"]["lr"] = 0.015 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "domainnet":
        # config["optimizer"]["lr_param"]["lr"] = 0.015 # optimal parameters
        config["network"]["params"]["class_num"] = 345
    else:
        raise ValueError(
            'Dataset cannot be recognized. Please define your own dataset here.'
        )

    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    train(config)
