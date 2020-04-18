import torch
from torch import nn
import numpy as np

# =============================================== RMSLE ================================================================
class RMSLELoss(nn.Module):
    def __init__(self):
        '''
        Root Mean Squared Log Error
        '''
        super().__init__()

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor, target):
        return np.sqrt(np.square( np.log(inTensor + 1) - np.log(target + 1) ).mean())