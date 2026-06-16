import torch
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex, BinaryAUROC
import torch.nn as nn
import torch.nn.functional as F

def dice(preds, targets):
    """
    expects 2 torch tensor inputs: preds and targets
    preds expeted binary prediction
    returns scalar value output dice coeff of the two
    """
    preds = preds.float()
    targets = targets.float()
    print(preds.shape, targets.shape)  # add this
    intersection = (preds * targets).sum()
    return (2 * (intersection) / (preds.sum() + targets.sum() + 1e-12)).item()

def jaccard(preds, targets):
    """"
    wrapper for torch binaryjaccerdinex, return jaccard of preds and 
    targets (two tensors). Assumes preds is binary prediction
    """
    preds = preds.float()
    targets = targets.float()
    metric = BinaryJaccardIndex().to(preds.device)
    return metric(preds, targets)

def auc(preds, targets):
    """"
    Returns auc given tensor inputs of labels
    and probability values (preds). Expects preds is sigmoid value
    (not logits or binary prediction)
    """
    binary_auroc = BinaryAUROC().to(preds.device)
    return binary_auroc(preds, targets)

def bacc(preds, targets):
    """"
    return balanced acc of preds and 
    targets (two tensors). Assumes preds is binary prediction
    """
    preds = preds.float()
    targets = targets.float()
    num_tp = ((preds == 1) & (targets == 1)).sum().float()
    num_tn = ((preds == 0) & (targets == 0)).sum().float()
    num_p = (targets == 1).sum().float()
    num_n = (targets == 0).sum().float()
    return ((num_tp / num_p) + (num_tn / num_n)) / 2


# need callable nn module for loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice
    

class DiceBCELoss(nn.Module):
    def __init__(self, lam_bce=1.0, lam_dice=1.0, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.lam_bce = lam_bce
        self.lam_dice = lam_dice
        self.smooth = smooth
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # 1. Compute BCE Loss
        bce_loss = self.bce_with_logits(inputs, targets)

        # 2. Convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # 3. Compute Dice Loss
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score

        # 4. Return weighted combined loss
        return (self.lam_bce * bce_loss) + (self.lam_dice * dice_loss)



