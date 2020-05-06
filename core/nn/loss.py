import numpy as np

import torch
from torch import nn

# =============================================== L1 NORM =====================================================
def l1_norm_error(source, candidate):

    error = np.abs(source - candidate)
    source[source == 0] = 1e-30  # add for numerical stability
    error = error / source       # compute the percentage
    error = error.mean()
    return error

# =============================================== RMSLE  =====================================================
def rmsle_error(source, candidate):
    candidate += 1e-30
    error = np.log10((source + 1) / (candidate + 1))
    error = error * error
    error = error.mean()
    error = np.sqrt(error)

    return error

# =============================================== GRADIENT SMOOTH =====================================================
class GradientSmoothLoss(nn.Module):
    def __init__(self, refGrad, future, decayFunc = None):
        '''
        Function that minimizes the rate of change of a time series prediction,
        as the times evolves. It tries to give a desired "shape".

        :param refGrad:   the maximum gradient that is used for scaling
        :param future:    number of future predictions in the timeseries
        :param decayFunc: decay function for weights (the weights decrease as time increases, such that the last
                            timestamps will have a smoother rate of change)
        '''

        super().__init__()
        self.future  = future
        self.refGrad = refGrad

        # compute decay weights
        decay = np.linspace(0, 1, future)
        decay = self.__linear_decay(decay) if decayFunc is None \
                                            else decayFunc(decay)
        decay = torch.from_numpy(decay)

        self.decay    = decay * refGrad

    # =============================================== LINEAR DECAY =====================================================
    def __linear_decay(self, linSpace):
        return 0.8 - linSpace * 0.5

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor, clampVal = 0.25):
        '''
        :param inTensor: input tensor on which to apply the loss
        :param clampVal: clamp errors before averaging for better stability
        :return:
        '''

        self.decay = self.decay.to(inTensor.device)

        gradOut = inTensor[:, 1:] - inTensor[:, :-1]
        gradOut = gradOut.abs() - self.decay
        gradOut = torch.clamp(gradOut, min=0, max=clampVal)
        gradOut = gradOut.mean()

        return gradOut


