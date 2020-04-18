import os

import importlib
import torch.optim
import numpy as np

from core.nn       import BaseTrainer
from core.networks import BasicRecurrentPredictor
from core.nn.loss import RMSLELoss
from torch.nn import MSELoss


class ReccurentTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # ================================================== BUILD =========================================================
    def build(self):

        self._init_models()
        self._init_optimizers()
        self._init_losses()
        self._init_eval_losses()
        self._init_schedulers()

        self._init_training_state(self.cfg.loadStep)
        return self

    # ================================================== MODEL =========================================================
    def _init_models(self):
        model = BasicRecurrentPredictor(**self.cfg.modelKwargs).build()
        model.to(self.cfg.toDevice)
        model.train()
        self.register_model(model, 'model')

    # ================================================== OPTIMIZER =====================================================
    def _init_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr': self.cfg.baseLr}]

        optimizer = getattr(torch.optim, self.cfg.optimizerName)(params, **self.cfg.optimizerKwargs)
        self.register_optimizer(optimizer, 'optimizer')

    # ================================================== INIT SCHEDULERS ===============================================
    def _init_schedulers(self):
        if self.cfg.schedName is not None:
            schedPack = importlib.import_module(self.cfg.schedPack)
            self.register_scheduler(getattr(schedPack, self.cfg.schedName)\
                                    (self.optimizer, **self.cfg.schedKwargs), 'scheduler')

    # ================================================== INIT LOSSES ===================================================
    def _init_losses(self):
        lossPack = importlib.import_module(self.cfg.lossPack)
        loss     = getattr(lossPack, self.cfg.lossName)(**self.cfg.lossKwargs)

        self.register_loss([loss] , ['loss'])
    # =============================================== INIT EVAL LOSSES =================================================
    def _init_eval_losses(self):
        lossDict = {}
        for lPack, lType in zip(self.cfg.lossEvalPack, self.cfg.lossEvalName):
            lossEvalPack    = importlib.import_module(lPack)
            evalLoss        = getattr(lossEvalPack, lType)()
            lossDict[lType] = evalLoss

        self.evalLossDict = lossDict

    # ================================================== INIT TRAINING PHASE ===========================================
    def _init_training_state(self, loadStep):
        # create experiment save dict
        if not os.path.exists(self.cfg.saveDir):
            os.makedirs(self.cfg.saveDir)

        if loadStep == 0:
            self.cfg.save()

        # init network weights
        if loadStep > 0:
            self.load_networks(loadStep)
        else:
            # note, the LSTM cell is initialized automatically,
            self._weightInit.init_weights(self.model, 'kaiming_normal_', {})

        # load optimizer state
        if loadStep > 0:
            self.load_optimizers(loadStep)

        # update the last epoch of  the scheduler to resume its value
        if loadStep > 0 and hasattr(self, 'scheduler'):
            self.scheduler.last_epoch = (loadStep + 1)

    # =============================================== OPTIMIZE PARAMETERS ==============================================
    def optimize_parameters(self, step, *args):
        data   = args[0].to(self.cfg.toDevice)
        labels = args[1].to(self.cfg.toDevice)

        self.optimizer.zero_grad()

        # process batch
        res            = self.model(data, target = labels)
        self.lossValue = self.loss(res, labels)
        self.lossValue.backward()

        # update pytorch
        self.optimizer.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        self.model.zero_grad()
    # =============================================== EVALUATE =========================================================
    def evaluate(self, testData, scalers, loss = MSELoss()):
        # TODO: Refactor
        evalVals = {key:np.zeros((1,2)) for key  in testData.keys()}

        for key, data in testData.items():
            obsTime  = self.cfg.winSize - self.cfg.predSize
            predTime = data.shape[0] - obsTime

            with torch.no_grad():
                res  = self.model(data[:obsTime,:].unsqueeze(0), future = predTime)
                res = scalers[key].inverse_transform(res[0].cpu())
                lbl = scalers[key].inverse_transform(data[obsTime:,:].cpu())
                print(lbl.shape, res.shape)
                lossVal  = loss(res, lbl)
                evalVals[key] = lossVal

        evalVals = {key:np.array(val).mean() for key, val in evalVals.items()}

        return evalVals
    # ================================================== GET LRs =======================================================
    def get_lr(self):
        return {'Model': float(self.optimizer.param_groups[0]['lr'])}


if __name__ == '__main__':
    pass