import torch
import numpy as np
import torch.nn as nn

from src.core.networks.mlp import MLP
from src.core.utils import Configurable

# =============================================== BASIC RECCURENT PRED =================================================
class BasicRecurrentPredictor(nn.Module, Configurable):
    def __init__(self, chNo, hidChNo = 128, future = 20, **kwargs):
        '''
        A simple reccurent predictor usgin a Recurrent Cell (LSTM)
            RECCURENT CELL ->  MLP -> output

        :param chNo: number of features
        :param hidChNo: number of hidden embedding size
        :param future: number of feature moments to predict
        :param kwargs: see internal config for more details
        '''

        super().__init__()
        self.chNo    = chNo
        self.hidChNo = hidChNo
        self.future  = future

        self.build_external_config()
        self._build_internal_config(**kwargs)

    # ===============================================  INTERNAL CONFIG =================================================
    @Configurable.internal_config()
    def _build_internal_config(self, **kwargs):
        ''' Only modify this parameters if you know what you are doing'''
        return {

            # used during training to apply randomly replace prediction with label
            'teacherProb' : 0.5,

            # rnn cell type and depth
            'rnnCell'    : 'LSTMCell',
            'rnnNoCells' : 2,

            # multi layer perceptron that is applied to the output of rnn
            'mlpLayerCfg' : [64,64],    # layer hidden dims
            'mlpActiv'    : 'Tanh',     # inner activation of the mlp
            'dropRate'    : None,       # dropout rate for each layer of mlp
            'normType' : ' BatchNorm1d',

            'mlpActivLast' : None,       # note that every timestamp in the sequence will be activated too
            'returnFullSeq': False
        }
    # =============================================== BUILD ============================================================
    def build(self):
        self.rnnModel = self.__get_rnn_model()
        self.mlpModel = self.__get_mlp()

        return self

    # =============================================== RNN ==============================================================
    def __get_rnn_model(self):
        rnnCell   = getattr(torch.nn, self.rnnCell)

        rnnModules = nn.ModuleList()
        for i in range(self.rnnNoCells):
            prevDepth = self.chNo if i == 0 else self.hidChNo
            rnnModules.append(rnnCell(prevDepth, self.hidChNo))

        return rnnModules

    # =============================================== MLP ==============================================================
    def __get_mlp(self):
        return MLP(self.hidChNo, self.chNo,    layerCfg = self.mlpLayerCfg,
                    activ    = self.mlpActiv,   activLast  = self.mlpActivLast,
                    dropRate = self.dropRate, normType = self.normType).build()

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor, future=0, target=None, teacherProb = None):
        '''

        :param inTensor:    input tensor [batch, moments, features]
        :param future:      no of future moments to predict
        :param target:      used during training, label
        :param teacherProb: used during training for applying teacher method (replacing intermediate
                            predictions with labels)
        :return:
        '''

        future      = self.future if future == 0 else future
        teacherProb = self.teacherProb if teacherProb is None else teacherProb
        outputs     = []

        # initializa hidden state/cell
        zeros     = lambda : torch.zeros(inTensor.shape[0], self.hidChNo, device = inTensor.device).float()
        hidState  = [zeros() for _ in range(len(self.rnnModel))]
        cellState = [zeros() for _ in range(len(self.rnnModel))]

        # embed features
        for moment in range(inTensor.shape[1]):

            # forward pass each cell
            for i, r in enumerate(self.rnnModel):
                inp = inTensor[:,moment,:] if i == 0 else hidState[i-1]
                hidState[i], cellState[i] = r(inp, (hidState[i], cellState[i]))

                output   = self.mlpModel(hidState[-i])
            if self.returnFullSeq:
                outputs += [output]

        # make feature predictions
        for moment in range(future):
            # teacher enforce
            if target is not None and np.random.random() > teacherProb:
                output = target[:, [moment]]

            # forward pass each cell
            for i, r in enumerate(self.rnnModel):
                inp = output.squeeze(1) if i == 0 else hidState[i-1]
                hidState[i], cellState[i] = r(inp, (hidState[i], cellState[i]))

            output   = self.mlpModel(hidState[i])
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    model = BasicReccurentPredictor().build()
    source = torch.randn((3, 10, 2))
    label = torch.randn((3, 20, 2))
    # seq(source)
    res = model(source, 20, label, 0.2)
    print(res.shape)