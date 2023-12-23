import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pandas as pd
import os
from utils.evaluate_12ECG_score import load_weights

# weights the loss based on the scoring matrix
# this loss class applies sigmoid, so it should be used in place of BCEWithLogitsLoss
class Confusion_Weighted_BCELoss(_Loss):
    def __init__(self, device):
        super(Confusion_Weighted_BCELoss, self).__init__()
        self.confusion_weight = confusion_weights(device)
        self.pos_weight = torch.ones(27, dtype=torch.float).to(device)
        print(self.confusion_weight)
    def forward(self, outputs, labels):
        # get confusion_weight for current labels
        conf_weight = torch.sum(labels.unsqueeze(2) * (1 - self.confusion_weight).unsqueeze(0), 1)
        num_labels = torch.sum(labels, 1, keepdim=True)
        # normalization in terms of num_labels (set conf weight 0.5 for all-negative labeled samples)
        conf_weight[num_labels.squeeze() == 0] = 0.5
        num_labels[num_labels == 0] = 1
        conf_weight = conf_weight / num_labels

        loss = -torch.mean(self.pos_weight * labels * F.logsigmoid(outputs) + conf_weight * (1 - labels) * F.logsigmoid(-outputs))

        return loss

class Challenge_Loss(_Loss):
    def __init__(self, device):
        super(Challenge_Loss, self).__init__()
        self.confusion_weight = confusion_weights()
        self.pos_weight = torch.ones(27, dtype=torch.float).to(device)
    def forward(self, outputs, labels):
        num_labels = torch.sum(labels, 1, keepdim=True)
        # loss = -torch.mean(self.pos_weight * labels * (self.confusion_weight @ F.logsigmoid(outputs).T).T)
        loss = -torch.mean(self.pos_weight * labels * F.logsigmoid(outputs) + ((1 - self.confusion_weight) @ ((1 - labels) * F.logsigmoid(-outputs)).T).T/(27-num_labels))
        return loss

class Combined_Losses(_Loss):
    def __init__(self, losses, weights):
        super(Combined_Losses, self).__init__()
        self.losses = losses
        self.weights = weights
    def forward(self, outputs, labels):
        total_loss = 0
        for i, loss in enumerate(self.losses):
            total_loss += loss(outputs, labels) * self.weights[i]
        return total_loss

def negative_over_positive_weights(device):
    dx_mapping_df = pd.read_csv(os.path.join('utils', 'dx_mapping_scored.csv'))
    pos = torch.tensor(dx_mapping_df['Total'])
    pos_weight = (torch.sum(pos)-pos)/pos
    pos_weight = pos_weight.to(device)
    return pos_weight

def confusion_weights(device):
    return torch.tensor(load_weights(os.path.join('utils', 'weights.csv'))[1], dtype=torch.float).to(device)