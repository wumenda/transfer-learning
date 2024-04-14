# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in train.py
"""
import math
import sys
from typing import Iterable
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

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
        losses = source_loss + 10 * mmd_loss
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
