import logging
import torch
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam


def create_optimizer(args, parameter_list):
    optimizer = Adam(parameter_list, lr=args.lr)
    steps = [int(step) for step in args.steps.split(",")]  # 通过逗号分隔的多个步数
    lr_scheduler = MultiStepLR(optimizer, milestones=steps, gamma=args.gamma)
    return optimizer, lr_scheduler
