import torch
from core.utils import misc
from experiments.simple_prediction.config import Config

if __name__ == '__main__':
    path      = 'assets/checkpoints/rnnPred_mse_big_lr/'
    modelPath =
    config = Config.load(path)

    misc.load_networks()