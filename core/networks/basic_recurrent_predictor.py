import torch
import numpy as np
import torch.nn as nn

from .mlp import MLP

# =============================================== BASIC RECCURENT PRED =================================================
class BasicRecurrentPredictor(nn.Module):
    def __init__(self, chNo, hidChNo = 128, future = 20, **kwargs):
        '''
        A  reccurent predictor composed from a Recurrent Cell and a MLP.

        :param chNo          : number of input features
        :param hidChNo       : number of hidden embedding size of rnn cell. Same hidden size is considered for all cells
        :param future        : number of future moments to predict

        :param kwargs        :
        :param teacherProb   : during training randomly replace prediction with label
        :param rnnCell       : type of RNN cell [LSTMCell, GRUCell]
        :param rnnNoCells    : number of RNN cells
        :param mlpLayerCfg   : list layer config
        :param dropRate      : dropout rate applied on each layer of the MLP
        :param normType      : normalization type for MLP
        :param mlpActivLast  : last layer activation of MLP
        :param returnFullSeq : return the encoded sequence the the future prediction
        '''

        super().__init__()
        self.chNo    = chNo
        self.hidChNo = hidChNo
        self.future  = future

        self.__hyper_parameters(**kwargs)

    # ===============================================  INTERNAL CONFIG =================================================
    def __hyper_parameters(self, **kwargs):
        ''' This method sets up the hyperparameters. Default values are provided.'''

        # used during training to randomly replace prediction with label
        self.teacherProb   = kwargs.get('teacherProb', 0.5)
        self.returnFullSeq = kwargs.get('returnFullSeq', False)

        # RNN config
        self.rnnCell       = kwargs.get('rnnCell',   'LSTMCell')
        self.rnnNoCells    = kwargs.get('rnnNoCells', 2)

        # MLP config
        self.mlpLayerCfg   = kwargs.get('mlpLayerCfg',  [64,64])
        self.mlpActiv      = kwargs.get('mlpActiv',     'Tanh')
        self.dropRate      = kwargs.get('dropRate',      None)
        self.normType      = kwargs.get('normType',      None)
        self.mlpActivLast  = kwargs.get('mlpActivLast',  None)

    # =============================================== BUILD ============================================================
    def build(self):
        '''
        Build the layer configuration of the components.
        :return: self
        '''

        self.rnnModel = self.__get_rnn_model()
        self.mlpModel = self.__get_mlp()

        return self

    # =============================================== RNN ==============================================================
    def __get_rnn_model(self):
        '''
        Compose the list of RNN cells.
        :return: module list
        '''
        rnnCell   = getattr(torch.nn, self.rnnCell)

        rnnModules = nn.ModuleList()
        for i in range(self.rnnNoCells):
            prevDepth = self.chNo if i == 0 else self.hidChNo
            rnnModules.append(rnnCell(prevDepth, self.hidChNo))

        return rnnModules

    # =============================================== MLP ==============================================================
    def __get_mlp(self):
        return MLP(self.hidChNo, self.chNo,     layerCfg  = self.mlpLayerCfg,
                    activ    = self.mlpActiv,   activLast = self.mlpActivLast,
                    dropRate = self.dropRate,   normType  = self.normType).build()

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor, future=0, target=None, teacherProb = None):
        '''

        :param inTensor:    input tensor [batch, moments, features]
        :param future:      number of future moments to predict
        :param target:      used during training, randomly replace future prediction with label
        :param teacherProb: used during training for applying teacher method (replacing intermediate
                            predictions with labels)
        :return:
        '''

        future      = self.future if future == 0 else future
        teacherProb = self.teacherProb if teacherProb is None else teacherProb
        outputs     = []

        # initialize hidden state/cell
        zeros     = lambda : torch.zeros(inTensor.shape[0], self.hidChNo, device = inTensor.device).float()
        hidState  = [zeros() for _ in range(len(self.rnnModel))]

        if self.rnnCell == 'LSTMCell':
            cellState = [zeros() for _ in range(len(self.rnnModel))]

        # embed features
        for moment in range(inTensor.shape[1]):

            # forward pass each cell
            for i, r in enumerate(self.rnnModel):
                inp = inTensor[:,moment,:] if i == 0 else hidState[i-1]

                if self.rnnCell == 'LSTMCell':
                    hidState[i], cellState[i] = r(inp, (hidState[i], cellState[i]))
                else: # GRU cell has not cell state
                    hidState[i] = r(inp,hidState[i])

            output   = self.mlpModel(hidState[-i])
            if self.returnFullSeq:
                outputs += [output]


        # make future predictions
        for moment in range(future):
            # teacher enforce
            if target is not None and np.random.random() < teacherProb:
                output = target[:, moment]

            # forward pass each cell
            for i, r in enumerate(self.rnnModel):
                inp = output if i == 0 else hidState[i-1]
                if self.rnnCell == 'LSTMCell':
                    hidState[i], cellState[i] = r(inp, (hidState[i], cellState[i]))
                else: # GRU cell has not cell state
                    hidState[i] = r(inp, hidState[i])

            output   = self.mlpModel(hidState[i])
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

# =============================================== DUMMY TEST =====================================================
if __name__ == '__main__':
    model = BasicRecurrentPredictor(
            chNo = 1,
            hidChNo = 128,
            future = 0,
            teacherProb = 0.5,
            # rnn cell type and depth
            rnnCell    = 'LSTMCell',
            rnnNoCells = 1,

            # multi layer perceptron that is applied to the output of rnn
            mlpLayerCfg   = [64,64],    # layer hidden dims
            mlpActiv      = 'ReLU',     # inner activation of the mlp
            dropRate      = None,       # dropout rate for each layer of mlp
            normType      = None,
            mlpActivLast  = None,       # note that every timestamp in the sequence will be activated too
            returnFullSeq = True).build()

    source = torch.randn((210, 8, 1))

    res = model(source[:,:4], 4, source[:, 4:], 0.0)
    print(res.shape)
