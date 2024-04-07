import logging
import torch
import pickle
import torch.optim as optim
from sklearn.model_selection import train_test_split


def create_optimizer(args, parameter_list):
    optimizer = optim.Adam(parameter_list, lr=args.lr)
    steps = int(args.steps.split(",")[0])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
    return optimizer, lr_scheduler
