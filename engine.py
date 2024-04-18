# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in train.py
"""
import copy
import math
import sys
from typing import Iterable
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from criterion import MK_MMD, domain_adaptation_loss_mmd, mmd_rbf, coral
from util.mectric import accuracy
from util.plot import draw_matrix
from util.t_sne import plot_sne
from data.cwru import __cwru_class__


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    lr_scheduler=None,
):
    model.train()
    criterion.train()
    losses_list = []
    acc_list = []
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        feature, outputs = model(samples)
        losses = criterion(outputs, targets)

        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 记录损失和准确率
        target_acc = accuracy(outputs, targets)
        acc_list.append(target_acc)  # 将准确率记录下来
        losses_list.append(losses.cpu().item())
    return acc_list, losses_list


@torch.no_grad()
def evaluate(model, data_loader, device, matrix_Savepath, tsne_savepath):
    model.eval()
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature, outputs = model(samples)
        _, predict = torch.max(outputs.cpu(), 1)
        target_acc = accuracy(outputs, targets)
        draw_matrix(predict, targets.cpu(), __cwru_class__, matrix_Savepath)
        plot_sne(feature.cpu().numpy(), targets.cpu().numpy(), tsne_savepath)
    return target_acc


@torch.no_grad()
def evaluate_without_plot(model, data_loader, device):
    model.eval()
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature, outputs = model(samples)
        _, predict = torch.max(outputs.cpu(), 1)
        target_acc = accuracy(outputs, targets)
    return target_acc


def train_one_epoch_transfer(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    source_loader: Iterable,
    target_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    for batch_idx, (
        (source_data, source_labels),
        (target_data, target_labels),
    ) in enumerate(zip(source_loader, target_loader)):
        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data, target_labels = target_data.to(device), target_labels.to(device)
        # 将源域数据输入模型
        source_features, source_outputs = model(source_data)
        # 将目标域数据输入模型
        target_features, target_outputs = model(target_data)
        optimizer.zero_grad()
        # 计算源域任务损失
        source_loss = criterion(source_outputs, source_labels)
        # 计算总体损失
        losses = source_loss
        # 反向传播和优化
        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # 记录损失和准确率
        target_acc = accuracy(target_outputs, target_labels)


def train_one_epoch_with_mmd(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    source_loader: Iterable,
    target_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_scheduler,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    losses_list = []
    acc_list = []
    cls_loss_list = []
    adver_loss_list = []
    for batch_idx, (
        (source_data, source_labels),
        (target_data, target_labels),
    ) in enumerate(zip(source_loader, target_loader)):
        # lr_scheduler.step()
        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data, target_labels = target_data.to(device), target_labels.to(device)
        # 将源域数据输入模型
        source_features, source_outputs = model(source_data)
        # 将目标域数据输入模型
        target_features, target_outputs = model(target_data)
        optimizer.zero_grad()
        # 计算源域任务损失
        source_loss = criterion(source_outputs, source_labels)
        # 计算领域自适应损失
        mmd_loss = mmd_rbf(source_features, target_features)
        # 计算总体损失
        losses = source_loss + mmd_loss
        # 反向传播和优化
        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # # 记录损失和准确率
        acc = accuracy(target_outputs, target_labels)
        acc_list.append(acc)  # 将准确率记录下来
        losses_list.append(losses.cpu().item())
        cls_loss_list.append(source_loss.cpu().item())
        adver_loss_list.append(mmd_loss.cpu().item())
    return acc_list, losses_list, cls_loss_list, adver_loss_list


def train_one_epoch_on_gan(
    feature_net: torch.nn.Module,
    classifier: torch.nn.Module,
    discriminator: torch.nn.Module,
    cls_criterion: torch.nn.Module,
    adver_criterion: torch.nn.Module,
    source_loader: Iterable,
    target_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    feature_net.train()
    classifier.train()
    discriminator.train()
    cls_criterion.train()
    adver_criterion.train()
    losses_list = []
    acc_list = []
    cls_loss_list = []
    adver_loss_list = []
    for batch_idx, (
        (source_data, source_labels),
        (target_data, target_labels),
    ) in enumerate(zip(source_loader, target_loader)):
        if len(source_labels) != len(target_labels):
            break
        batch_num = source_labels.size(0)
        inputs = torch.cat((source_data, target_data), dim=0)
        labels = torch.cat((source_labels, target_labels), dim=0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        source_labels = source_labels.to(device)
        target_labels = target_labels.to(device)

        # 构造域标签
        domain_label_source = torch.ones(batch_num).float().to(device)
        domain_label_target = torch.zeros(batch_num).float().to(device)
        domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)

        # 将源域数据输入模型
        logits = feature_net(inputs)
        outputs = classifier(logits)
        domain_outputs = discriminator(logits)

        optimizer.zero_grad()
        # 计算源域任务损失
        cls_loss = cls_criterion(outputs.narrow(0, 0, batch_num), source_labels)
        # 计算域混淆损失
        adver_loss = adver_criterion(domain_outputs.squeeze(), domain_label)

        # 计算总体损失
        losses = cls_loss + 10 * adver_loss
        # 反向传播和优化
        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)
        losses.backward()
        optimizer.step()

        # # 记录损失和准确率
        acc = accuracy(outputs.narrow(0, batch_num, batch_num), target_labels)
        acc_list.append(acc)  # 将准确率记录下来
        losses_list.append(losses.cpu().item())
        cls_loss_list.append(cls_loss.cpu().item())
        adver_loss_list.append(adver_loss.cpu().item())
    return acc_list, losses_list, cls_loss_list, adver_loss_list


def train_one_epoch_on_gan_with_mmd(
    feature_net: torch.nn.Module,
    classifier: torch.nn.Module,
    discriminator: torch.nn.Module,
    cls_criterion: torch.nn.Module,
    adver_criterion: torch.nn.Module,
    source_loader: Iterable,
    target_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    feature_net.train()
    classifier.train()
    discriminator.train()
    cls_criterion.train()
    adver_criterion.train()
    losses_list = []
    acc_list = []
    cls_loss_list = []
    adver_loss_list = []
    for batch_idx, (
        (source_data, source_labels),
        (target_data, target_labels),
    ) in enumerate(zip(source_loader, target_loader)):
        if len(source_labels) != len(target_labels):
            break
        batch_num = source_labels.size(0)
        inputs = torch.cat((source_data, target_data), dim=0)
        labels = torch.cat((source_labels, target_labels), dim=0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        source_labels = source_labels.to(device)
        target_labels = target_labels.to(device)

        # 构造域标签
        domain_label_source = torch.ones(batch_num).float().to(device)
        domain_label_target = torch.zeros(batch_num).float().to(device)
        domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)

        # 将源域数据输入模型
        logits = feature_net(inputs)
        outputs = classifier(logits)
        domain_outputs = discriminator(logits)

        optimizer.zero_grad()
        # 计算源域任务损失
        cls_loss = cls_criterion(outputs.narrow(0, 0, batch_num), source_labels)
        # 计算域混淆损失
        adver_loss = adver_criterion(domain_outputs.squeeze(), domain_label)
        # 计算mmd损失
        mmd_loss = mmd_rbf(
            logits.narrow(0, 0, batch_num), logits.narrow(0, batch_num, batch_num)
        )

        # 计算总体损失
        losses = cls_loss + 10 * adver_loss + 10 * mmd_loss
        # 反向传播和优化
        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)
        losses.backward()
        optimizer.step()
        # 记录损失和准确率
        acc = accuracy(outputs.narrow(0, batch_num, batch_num), target_labels)
        # # 记录损失和准确率
        acc = accuracy(outputs.narrow(0, batch_num, batch_num), target_labels)
        acc_list.append(acc)  # 将准确率记录下来
        losses_list.append(losses.cpu().item())
        cls_loss_list.append(cls_loss.cpu().item())
        adver_loss_list.append(adver_loss.cpu().item())
    return acc_list, losses_list, cls_loss_list, adver_loss_list


def train_one_epoch_on_gan_with_mmd_pseudo_label(
    feature_net: torch.nn.Module,
    classifier: torch.nn.Module,
    discriminator: torch.nn.Module,
    pre_train_feature_net: torch.nn.Module,
    pre_train_classifier: torch.nn.Module,
    cls_criterion: torch.nn.Module,
    adver_criterion: torch.nn.Module,
    source_loader: Iterable,
    target_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    pre_train_feature_net.eval()
    pre_train_classifier.eval()
    feature_net.train()
    classifier.train()
    discriminator.train()
    cls_criterion.train()
    adver_criterion.train()
    losses_list = []
    acc_list = []
    cls_loss_list = []
    adver_loss_list = []
    for batch_idx, (
        (source_data, source_labels),
        (target_data, target_labels),
    ) in enumerate(zip(source_loader, target_loader)):
        if len(source_labels) != len(target_labels):
            break
        batch_num = source_labels.size(0)
        inputs = torch.cat((source_data, target_data), dim=0)
        labels = torch.cat((source_labels, target_labels), dim=0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        target_data = target_data.to(device)
        source_labels = source_labels.to(device)
        target_labels = target_labels.to(device)

        # 生成伪标签
        pseudo_label = F.softmax(
            pre_train_classifier(pre_train_feature_net(target_data)), dim=1
        )

        # 构造域标签
        domain_label_source = torch.ones(batch_num).float().to(device)
        domain_label_target = torch.zeros(batch_num).float().to(device)
        domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)

        # 将数据输入模型
        logits = feature_net(inputs)
        outputs = classifier(logits)
        domain_outputs = discriminator(logits)

        optimizer.zero_grad()
        # 计算源域任务损失
        cls_loss = cls_criterion(outputs.narrow(0, 0, batch_num), source_labels)
        # 计算域混淆损失
        adver_loss = adver_criterion(domain_outputs.squeeze(), domain_label)
        # 计算mmd损失
        mmd_loss = mmd_rbf(
            logits.narrow(0, 0, batch_num), logits.narrow(0, batch_num, batch_num)
        )
        # 计算伪标签损失
        pseudo_loss = cls_criterion(
            outputs.narrow(0, batch_num, batch_num), pseudo_label
        )

        # 计算总体损失
        losses = cls_loss + pseudo_loss + mmd_loss + adver_loss
        # 反向传播和优化
        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)
        losses.backward()
        optimizer.step()
        # 记录损失和准确率
        acc = accuracy(outputs.narrow(0, batch_num, batch_num), target_labels)
        # # 记录损失和准确率
        acc = accuracy(outputs.narrow(0, batch_num, batch_num), target_labels)
        acc_list.append(acc)  # 将准确率记录下来
        losses_list.append(losses.cpu().item())
        cls_loss_list.append(cls_loss.cpu().item())
        adver_loss_list.append(adver_loss.cpu().item())
    return acc_list, losses_list, cls_loss_list, adver_loss_list


@torch.no_grad()
def evaluate_gan(
    feature_net, classifier, data_loader, device, matrix_Savepath, tsne_savepath
):
    feature_net.eval()
    classifier.eval()
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature = feature_net(samples)
        outputs = classifier(feature)
        _, predict = torch.max(outputs.cpu(), 1)
        draw_matrix(predict, targets.cpu(), __cwru_class__, matrix_Savepath)
        plot_sne(feature.cpu().numpy(), targets.cpu().numpy(), tsne_savepath)
        acc = accuracy(outputs, targets)
    return acc


def train_one_epoch_on_hct(
    cnn_feature_net,
    transformer_feature_net,
    classifier4cnn,
    classifier4transformer,
    classifier4merge,
    discriminator: torch.nn.Module,
    cls_criterion: torch.nn.Module,
    adver_criterion: torch.nn.Module,
    source_loader: Iterable,
    target_loader: Iterable,
    source_loader_fft: Iterable,
    target_loader_fft: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    cnn_feature_net.train()
    transformer_feature_net.train()
    classifier4cnn.train()
    classifier4transformer.train()
    classifier4merge.train()
    discriminator.train()
    cls_criterion.train()
    adver_criterion.train()
    losses_list = []
    acc_list = []
    cls_loss_list = []
    adver_loss_list = []
    for batch_idx, (
        (source_data, source_labels),
        (target_data, target_labels),
        (source_data_fft, source_labels_fft),
        (target_data_fft, target_labels_fft),
    ) in enumerate(
        zip(source_loader, target_loader, source_loader_fft, target_loader_fft)
    ):
        if len(source_labels) != len(target_labels):
            break
        batch_num = source_labels.size(0)

        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        source_data_fft = source_data_fft.to(device)
        source_labels_fft = source_labels_fft.to(device)
        target_data_fft = target_data_fft.to(device)
        target_labels_fft = target_labels_fft.to(device)

        # cnn分支
        source_logits_cnn = cnn_feature_net(source_data_fft)
        output_cnn = classifier4cnn(source_logits_cnn)
        cls_loss_cnn = cls_criterion(output_cnn, source_labels_fft)

        # transformer分支
        source_logits_transformer = transformer_feature_net(source_data_fft)
        output_transformer = classifier4transformer(source_logits_transformer)
        cls_loss_transformer = cls_criterion(output_transformer, source_labels_fft)

        # merge分支
        source_logits_merge = torch.cat(
            (source_logits_cnn, source_logits_transformer), dim=-1
        )
        output_merge = classifier4merge(source_logits_merge)
        cls_loss_merge = cls_criterion(output_merge, source_labels_fft)

        # 对抗学习域自适应分支
        domain_label_source = torch.ones(batch_num).float().to(device)
        domain_label_target = torch.zeros(batch_num).float().to(device)
        domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)
        target_logits_cnn = cnn_feature_net(target_data_fft)
        target_logits_transformer = transformer_feature_net(target_data_fft)
        target_logits_merge = torch.cat(
            (target_logits_cnn, target_logits_transformer), dim=-1
        )
        domain_outputs = discriminator(
            torch.cat((source_logits_merge, target_logits_merge), dim=0)
        )
        adver_loss = adver_criterion(domain_outputs.squeeze(), domain_label)

        mmd_loss = mmd_rbf(source_logits_cnn, target_logits_cnn)

        optimizer.zero_grad()
        # 计算总体损失
        losses = (
            cls_loss_cnn + cls_loss_transformer + cls_loss_merge + adver_loss + mmd_loss
        )
        # 反向传播和优化
        if not math.isfinite(losses):
            print(f"Loss is :{losses}, stopping training")
            sys.exit(1)
        losses.backward()
        optimizer.step()
        # 记录损失和准确率
        acc = accuracy(output_merge, source_labels_fft)
        acc_list.append(acc)  # 将准确率记录下来
        losses_list.append(losses.cpu().item())
        cls_loss_list.append(cls_loss_merge.cpu().item())
        adver_loss_list.append(adver_loss.cpu().item())
    return acc_list, losses_list, cls_loss_list, adver_loss_list


@torch.no_grad()
def evaluate_merge(
    cnn_feature_net,
    transformer_feature_net,
    classifier4merge,
    data_loader,
    device,
    matrix_Savepath,
    tsne_savepath,
):
    cnn_feature_net.eval()
    transformer_feature_net.eval()
    classifier4merge.eval()
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature_cnn = cnn_feature_net(samples)
        feature_transformer = transformer_feature_net(samples)
        outputs_merge = classifier4merge(
            torch.cat((feature_cnn, feature_transformer), dim=-1)
        )
        _, predict = torch.max(outputs_merge.cpu(), 1)
        draw_matrix(predict, targets.cpu(), __cwru_class__, matrix_Savepath)
        plot_sne(feature_cnn.cpu().numpy(), targets.cpu().numpy(), tsne_savepath)
        acc = accuracy(outputs_merge, targets)
    return acc


@torch.no_grad()
def evaluate_fusion(
    cnn_feature_net,
    classifier4cnn,
    transformer_feature_net,
    classifier4transformer,
    classifier4merge,
    data_loader,
    device,
    matrix_Savepath,
    tsne_savepath,
):
    cnn_feature_net.eval()
    classifier4cnn.eval()
    transformer_feature_net.eval()
    classifier4transformer.eval()
    classifier4merge.eval()
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature_cnn = cnn_feature_net(samples)
        out_cnn = classifier4cnn(feature_cnn)
        feature_transformer = transformer_feature_net(samples)
        out_transformer = classifier4transformer(feature_transformer)
        outputs_merge = classifier4merge(
            torch.cat((feature_cnn, feature_transformer), dim=-1)
        )
        # 定义权重，这里假设权重是固定的
        weights = [0.3, 0.4, 0.3]  # 分别对应三个分支的权重
        sigmoid_layer = nn.Sigmoid()

        # 对三个分支的输出结果进行加权平均
        weighted_avg = (
            out_cnn * weights[0]
            + out_transformer * weights[1]
            + outputs_merge * weights[2]
        )
        _, predict = torch.max(weighted_avg.cpu(), 1)
        draw_matrix(predict, targets.cpu(), __cwru_class__, matrix_Savepath)
        plot_sne(feature_cnn.cpu().numpy(), targets.cpu().numpy(), tsne_savepath)
        acc = accuracy(outputs_merge, targets)
    return acc
