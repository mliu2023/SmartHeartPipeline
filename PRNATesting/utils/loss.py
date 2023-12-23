import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pandas as pd
from utils.evaluate_12ECG_score import load_weights

# weights the loss based on the scoring matrix
# this loss class applies sigmoid, so it should be used in place of BCEWithLogitsLoss
class Confusion_Weighted_BCELoss(_Loss):
    def __init__(self):
        super(Confusion_Weighted_BCELoss, self).__init__()
        self.confusion_weight = torch.tensor(load_weights("utils\\weights.csv")[1]).cuda()
        self.pos_weight = torch.ones(27, dtype=torch.float).cuda()
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
    def __init__(self):
        super(Challenge_Loss, self).__init__()
        self.confusion_weight = torch.tensor(load_weights("utils\\weights.csv")[1], dtype=torch.float).cuda()
        self.pos_weight = torch.ones(27, dtype=torch.float).cuda()
    def forward(self, outputs, labels):
        loss = -torch.mean(self.pos_weight * labels * (self.confusion_weight @ F.logsigmoid(outputs).T).T)
        return loss

class Combined_Loss(_Loss):
    def __init__(self, loss1, loss2, alpha=1, beta=1):
        super(Combined_Loss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha
        self.beta = beta
    def forward(self, outputs, labels):
        x = self.loss1(outputs, labels)
        y = self.loss2(outputs, labels)
        return self.alpha * x + self.beta * y

class Combined_Losses(_Loss):
    def __init__(self, losses, weights):
        super(Combined_Loss, self).__init__()
        self.losses = losses
        self.weights = weights
    def forward(self, outputs, labels):
        total_loss = 0
        for i, loss in enumerate(self.losses):
            total_loss += loss(outputs, labels) * self.weights[i]
        return total_loss

def negative_over_positive_weights():
    dx_mapping_df = pd.read_csv('utils\\dx_mapping_scored.csv')
    pos = torch.tensor(dx_mapping_df['Total'])
    # print(f'Number of samples for each class across the 6 datasets: {pos}')
    # print(f'Total number of samples across the 6 datasets: {torch.sum(pos)}')
    pos_weight = (torch.sum(pos)-pos)/pos
    pos_weight = pos_weight.cuda()
    # print(f'Weights for each class: {pos_weight}')
    return pos_weight