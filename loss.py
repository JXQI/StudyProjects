import torch.nn as nn
import torch.optim as optim


def loss(loss_name):
    if loss_name=="CrossEntropyLoss":
        L=nn.CrossEntropyLoss()