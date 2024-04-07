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
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(targets.data, 1)
        total += targets.size(0)
        correct += torch.sum(predicted == labels)
        acc = correct / total


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
    for i, (samples, targets) in enumerate(data_loader):
        targets = targets.to(device)
        samples = samples.to(device)
        feature, outputs = model(samples)

        _, labels = torch.max(targets.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += torch.sum(predicted == labels)
        acc = correct / total
        print("准确率:", acc.cpu().numpy())
        print("=============================")
        plot_sne(feature.cpu().numpy(), labels.cpu().numpy())
    return acc.cpu()
