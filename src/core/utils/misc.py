import os
import torch

# =============================================== LOAD NETWORKS ========================================================
def load_networks(path, model):
    '''
    Load the networks from the configured saveDir. Please consider transfering each network to the desired device.
    :param saveIdx: the index to load
    :return:
    '''

    loadFile = os.path.join(path)

    print('Load network %s' % loadFile)
    stateDict = torch.load(loadFile)

    if hasattr(stateDict, '_metadata'):
        del stateDict._metadata

    model.load_state_dict(stateDict)