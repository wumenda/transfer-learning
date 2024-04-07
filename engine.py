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

from criterion import MK_MMD, domain_adaptation_loss_mmd, mmd_rbf, coral
from util.t_sne import plot_sne


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
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
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
        _, predicted = torch.max(outputs, 1)
        labels = targets
        total += targets.size(0)
        correct += torch.sum(predicted == labels)
        acc = correct / total


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
    loss_list = []
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature, outputs = model(samples)
        losses = criterion(outputs, targets)
        loss_list.append(losses)

        labels = targets
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += torch.sum(predicted == labels)
        acc = correct / total
        # plot_sne(feature.cpu().numpy(), labels.cpu().numpy())
    return acc.cpu().item()


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
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
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
        # coral_loss = coral(source_features, target_features)

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
        # 记录损失和准确率
        _, predicted = torch.max(target_outputs, 1)
        labels = target_labels
        total += target_labels.size(0)
        correct += torch.sum(predicted == labels)
        acc = correct / total
