import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

# SCE loss
class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) # A=-4
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


# ANNLoss_K_2 is the loss when K=2
class ANNLoss_K_2(torch.nn.Module):
    def __init__(self, labels, num_classes, es=60, momentum=0.9):
        super(ANNLoss_K_2, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=False)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.es = es
        self.momentum = momentum
    # logits is the NN output for mixed feature, logits_ori is the NN output for original feature
    # labels_a is the original label, labels_b is the label for mixed data
    def forward(self, logits, logits_ori, labels_a, labels_b, index, nn_index, lam, epoch):
        pred = F.softmax(logits, dim=1)
        if epoch >self.es:
            lam=lam.cuda()
        if epoch < self.es:
            l1 = F.nll_loss(pred, labels_a, reduction='none')
            l2 = F.nll_loss(pred, labels_b, reduction='none')
        else:
            if epoch % 10 == 0:
                pred_ori_detach = F.softmax(logits_ori.detach(), dim=1)
                self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * pred_ori_detach
            l1 = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
            l2 = -torch.sum(torch.log(pred) * self.soft_labels[index[nn_index]], dim=1)

        loss = lam*l1+(1-lam)*l2
        return torch.mean(loss)