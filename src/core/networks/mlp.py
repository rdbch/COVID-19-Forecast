import torch
from torch import nn

from torch.nn import Linear, Dropout
from src.core.utils.configurable import Configurable



# ================================================== MLP ===============================================================
class MLP(nn.Module, Configurable):
    def __init__(self, inChNo=64, outChNo=None, layerCfg=(64, 64), **kwargs):
        '''
        Multi layer perceptron class
        :param inChNo: number of input features
        :param outChNo: number of output features (if None, use layerCfg[-1])
        :param layerCfg: a list containing the number of neurons for each layer
        :param kwargs: see internal config
        '''
        super().__init__()

        self.inChNo   = inChNo
        self.outChNo  = outChNo
        self.layerCfg = layerCfg

        self.build_external_config()
        self.build_internal_config(**kwargs)

    # =============================================== INTERNAL CONFIG ==================================================
    @Configurable.internal_config()
    def build_internal_config(self, **kwargs):
        return {
            'activ'     : 'Tanh', 'activKwargs'     : {},  # activation
            'activLast' : 'None', 'activLastKwargs' : {},  # last layer activation
            'norm'      : None,   'normKwargs'      : {},                 # normalization layer applied
            'dropRate'  : None,   'dropLayers'      : len(self.layerCfg)  # dropRate of Dropout, apply only on the last
                                                                          # dropLayers
            }

    # =============================================== BUILD ============================================================
    def build(self):
        self.model = self.__get_model()
        return self

    # =============================================== CHECK DROPOUT ====================================================
    def __use_droput(self, i):
        if self.dropRate is not None:
            if self.dropRate > 0:
               if i > (len(self.layerCfg) - self.dropLayers):
                   return True

        return False
    # =============================================== GET MODEl ========================================================
    def __get_model(self):
        model = nn.ModuleList()

        # get activation and normalization layer
        normLayer  = getattr(torch.nn, self.norm)  if self.norm is not None else None
        activLayer = getattr(torch.nn, self.activ) if self.activ is not None else None

        # only use bias when no normalization is used (redundant)
        useBias = True if normLayer is None else False

        # create layer structure
        for i, layerCfg in enumerate(self.layerCfg):

            prevChNo = self.inChNo if i == 0 else self.layerCfg[i-1]

            model.append(Linear(prevChNo, layerCfg, bias=useBias))
            if normLayer  is not None: model.append(normLayer(**self.normKwargs))
            if activLayer is not None: model.append(activLayer(**self.activKwargs))
            if self.__use_droput(i)  : model.append(Dropout(self.dropRate))

        # add last layer
        if self.outChNo is not None:
            activLastLayer = getattr(torch.nn, self.activLast) if self.activLast is not None else None
            prevChNo = self.layerCfg[-1] if len(self.layerCfg) > 0 else self.inChNo
            model.append(Linear(prevChNo, self.outChNo))
            if activLastLayer is not None: model.append(activLastLayer(**self.activLastKwargs))

        return model

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        for module in self.model:
            inTensor = module(inTensor)

        return inTensor
