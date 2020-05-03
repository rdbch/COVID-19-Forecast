import torch
from torch import nn

from torch.nn import Linear, Dropout

# ================================================== MLP ===============================================================
class MLP(nn.Module):
    def __init__(self, inChNo=64, outChNo=None, layerCfg=(64, 64), **kwargs):
        '''
        Multi layer perceptron class
        :param inChNo           : number of input features
        :param outChNo          : number of output features (if None, use layerCfg[-1])
        :param layerCfg         : a list containing the number of neurons for each layer
        :param kwargs
        :param activ            : activation used on the output of each layer
        :param activKwargs      : kwargs for activ
        :param activLast        : activation used on the last layer
        :param activLastKwargs  : kwargs for activLast
        :param normType         : normalization type (name from torch)
        :param normKwargs       : kwargs passed to normalization
        :param dropRate         : dropout rate
        :param dropLayers       : only apply dropout on the last dropLayers
        '''
        super().__init__()

        self.inChNo   = inChNo
        self.outChNo  = outChNo
        self.layerCfg = layerCfg

        self.__hyper_parameters(**kwargs)

    # =============================================== INTERNAL CONFIG ==================================================
    def __hyper_parameters(self, **kwargs):
        ''' This method sets up the hyperparameters. Default values are provided.'''

        # activation
        self.activ           = kwargs.get('activ',           'Tanh')
        self.activKwargs     = kwargs.get('activKwargs',     {})
        self.activLast       = kwargs.get('activLast',       None)
        self.activLastKwargs = kwargs.get('activLastKwargs', {})

        # normalization
        self.norm            = kwargs.get('normType',   None)
        self.normKwargs      = kwargs.get('normKwargs', {})

        # dropout
        self.dropRate        = kwargs.get('dropRate',   None)
        self.dropLayers      = kwargs.get('dropLayers', len(self.layerCfg))


    # =============================================== BUILD ============================================================
    def build(self):
        '''
        Build the layer configuration of the components.
        :return: self
        '''
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
        '''
        Construct the mlp layer configuration based on the set parameters.
        :return: ModuleList of layers
        '''
        model = nn.ModuleList()

        # get activation and normalization layer
        normLayer  = getattr(torch.nn, self.norm)  if self.norm is not None else None
        activLayer = getattr(torch.nn, self.activ) if self.activ is not None else None

        # only use bias when no normalization is used
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
        '''
        :param inTensor: input tensor
        :return: prediction
        '''
        for module in self.model:
            inTensor = module(inTensor)

        return inTensor
