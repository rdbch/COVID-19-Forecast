import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from core.data  import CovidDataset
from core.utils import VisdomClient
from experiments.simple_prediction.config import Config
from experiments.simple_prediction.trainer import ReccurentTrainer

# =============================================== ARGS =================================================================
argParse = argparse.ArgumentParser('Train Recurrent Predictor')
argParse.add_argument('-c', '--cfg', default = None, type=str, help='Load configuration')
argParse.add_argument('-n', '--name', default = 'rnnPred_mse_country_norm_small_lr_bn', type=str, help='Load configuration')

# =============================================== MAIN =================================================================
if __name__ == '__main__':

    # set for reproduction
    torch.manual_seed(53235)
    np.random.seed(53235)

    # get configs
    args = argParse.parse_args()
    if args.cfg is not None:
        cfg = Config.load(args.cfg)
    else:
        cfg = Config(args.name)

    # init visualizer
    vis  = VisdomClient(env=cfg.experimentName)
    trainEnd = pd.to_datetime('2020-03-25')
    data = CovidDataset('assets', windowSize=cfg.winSize,
                        predictSize=cfg.predSize, batchSize=cfg.batchSize,
                        device=cfg.toDevice, trainEnd=trainEnd)

    # build training model
    trainer = ReccurentTrainer(cfg)
    trainer.build()

    # visualization
    visSteps  = 100
    plotSteps = 10
    saveSteps = 500
    evalSteps = 500

    # ================================================== TRAINING LOOP =================================================
    for step in tqdm(range(cfg.loadStep+1, cfg.noSteps), initial=cfg.loadStep+1):
        trainData = data[step]

        trainer.optimize_parameters(step, *trainData)

        # plot loss evolution on evaluate dataset
        if step % plotSteps == 0:
            vis.plot_param(step, trainer.get_loss_values(), 'loss')
            vis.plot_param(step, trainer.get_lr(), 'lr')

        # save model
        if (step % saveSteps == 0 ) or (step == (cfg.noSteps - 1)):
            trainer.save_networks(step)
            trainer.save_optimizers(step)
            vis.save()

        # eval model
        if (step % evalSteps == 0 ) or (step == (cfg.noSteps - 1)):
            testData = data.get_test_data(transform=True)
            evalLoss = trainer.evaluate(testData, data.scalers)
            meanLoss = 0
            for key, value in evalLoss.items():
                meanLoss += value
            meanLoss /= len(evalLoss.keys())
            evalLoss['mean'] = meanLoss
            vis.plot_param(step, evalLoss, 'eval')