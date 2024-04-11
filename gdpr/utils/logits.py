import torch
from torch import nn

def logits_to_probs(y):
    y_probs = torch.softmax(y, dim=1) #batch
    return y_probs

def probs_to_max_classes(y_probs):
    y_max_values, y_max_indices  = torch.max(y_probs, dim=1)
    return y_max_values, y_max_indices

def logits_to_cross_entropy_loss(y, targets):
    print(y.shape)
    print(targets.shape)
    return nn.CrossEntropyLoss()(y, targets)