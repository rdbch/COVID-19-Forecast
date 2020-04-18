import os
from collections import OrderedDict

import torch

# ==================================== WEIGHT INITIALIZER ===========================================
class WeightInitializer:
    '''
    Utiility class for initializing the weights of a network.

    Usage example:
        weightInit = WeightInitializer()
        weightInit.init_weights(model, 'xavier_normal_', {'gain':0.02})

    '''

    def __init__(self, initType=None, kwargs={}):
        self.kwargs     = kwargs
        self.weightInit = None

        if initType is not None:
            if not hasattr(torch.nn, initType):
                raise NotImplementedError('Init method [%s] does not exist in torch.nn' % initType)
            self.weightInit = getattr(torch.nn.init, initType)

    # ===============================================  INIT WEIGHTS =================================
    def init_weights(self, model, weightInit=None, kwargs={}):
        '''
        Function called for initializeing the weights of a model
        :param model: pytorch model
        :param weightInit: init type (must be in torch.nn.init.*)
        :param kwargs: kwargs to be passed to the initialization function
        :return:
        '''

        if weightInit is not None:
            if not hasattr(torch.nn.init, weightInit):
                raise NotImplementedError('Init method %s not in torch.nn' % weightInit)
            self.weightInit = getattr(torch.nn.init, weightInit)

        self.kwargs = kwargs if kwargs != {} else self.kwargs

        model.apply(self._init_module)

    # =============================================== INIT MODULES ====================================================
    def _init_module(self, module):
        '''
        Internal function which is applied to every module in a network

        :param module: model to be applied to
        '''

        className = module.__class__.__name__

        # init conv and linear layers
        if hasattr(module, 'weight') and ('Conv' in className or 'Linear' in className):
            self.weightInit(module.weight.data, **self.kwargs)

            # init biases
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)

        # init batch norm weight
        # only normal distribution applies.
        elif className.find('BatchNorm2d') != -1:
            torch.nn.init.constant_(module.weight.data, 1.0)
            torch.nn.init.constant_(module.bias.data, 0.0)

# ================================================== BASE TRAINER ======================================================
class BaseTrainer:
    def __init__(self):

        # list of names
        self.__modelNames     = []                        # list with model      names (the same as the attrs)
        self.__lossNames      = []                        # list with loss       names (the same as the attrs)
        self.__optimizerNames = []                        # list with optimizers names (the same as the attrs)
        self.__lrSchedNames   = []                        # list with scheduler  names (the same as the attrs)
        self.__dataNames      = []                        # list with datasets   names (the same as the attrs)

        self.__lossValues     = {}

        # utility weight initializer
        self._weightInit     = WeightInitializer()

    # =============================================== REGISTER MODEL ===================================================
    def register_model(self, model, name):
        '''
        Register a nn.Module as a model. It has a similar functionality like when a nn.Module is declared inside another.
        It also sets the attribute to self.
        :param model: a torh.nn.module model
        :param name: the name of the model
        :return:
        '''

        model = [model] if type(model) is not list else model
        name  = [name]  if type(name)  is not list else name
        assert len(model) == len(name), f'Please use the same length.Current: models: {len(model)} , names: {len(name)}'

        self.__assert_has_attrs(name, 'model')

        self.__modelNames += name
        for n, value in zip(name, model):
            setattr(self, n, value)

    # =============================================== REGISTER LOSS ====================================================
    def register_loss(self, loss, name):
        '''
        Register a nn.Module as a model. It has a similar functionality like when a nn.Module is declared inside another.
        It also sets the attribute to self.
        :param loss: a loss or a list of losses
        :param name: the name of the model
        :return:
        '''

        loss = [loss] if type(loss) not in [list, tuple] else loss
        name = [name] if type(name) not in [list, tuple] else name
        assert len(loss) == len(name), f'Please use the same length. Current: loss: {len(loss)},'\
                                       f' names: {len(name)}'

        self.__assert_has_attrs(name, 'loss')

        self.__lossNames += name
        for n, value in zip(name, loss):
            setattr(self, n, value)
            setattr(self, n + 'Value', 0)

    # =============================================== REGISTER OPTIMIZER ===============================================
    def register_optimizer(self, optimizer, name):
        '''
        Register a nn.Module as a model. It has a similar functionality like when a nn.Module is declared inside another.
        It also sets the attribute to self.
        :param scheduler: a torh.nn. scheduler
        :param name: the name of the model
        :return:
        '''
        optimizer = [optimizer] if type(optimizer) not in [list, tuple] else optimizer
        name      = [name]      if type(name)      not in [list, tuple] else name

        assert len(optimizer) == len(name), f'Please use the same length. Current: optimizer: {len(optimizer)},' \
                                            f' names: {len(name)}'

        self.__assert_has_attrs(name, 'optimizer')

        self.__optimizerNames += name
        for n, value in zip(name, optimizer):
            setattr(self, n, value)

    # =============================================== REGISTER OPTIMIZER ===============================================
    def register_scheduler(self, scheduler, name):
        '''
        Register a nn.Module as a model. It has a similar functionality like when a nn.Module is declared inside another.
        It also sets the attribute to self.
        :param scheduler: a torh.nn scheduler
        :param name: the name of the model
        :return:
        '''
        scheduler = [scheduler] if type(scheduler) not in [list, tuple] else scheduler
        names     = [name]      if type(name)      not in [list, tuple] else name

        assert len(scheduler) == len(names), f'Please use the same length. Current: scheduler: {len(scheduler)},' \
                                            f' names: {len(names)}'

        self.__assert_has_attrs(names, 'scheduler')

        self.__lrSchedNames += names
        for name, value in zip(names, scheduler):
            setattr(self, name, value)

    # =============================================== ASSERT HAS ATTR ==================================================
    def __assert_has_attrs(self, names, regType = None):
        '''
        Check if self has the attribute
        :param names: the list of names to check
        :param regType: assertion error name
        '''

        regType = regType if regType is not None else ''

        for name in names:
            assert not hasattr(self, name), f'Could not register {regType}. Self already has [{name}] attribute.'

    # =============================================== EVAL MODE ========================================================
    def eval_mode(self):
        '''
        Put every registered module in eval mode.
        :return:
        '''
        for name in self.__modelNames:
            model = getattr(self, name)
            model.eval()

    # =============================================== TRAIN MODE =======================================================
    def train_mode(self):
        '''
        Put all of the modules in evaluate mode.
        :return:
        '''
        for name in self.__modelNames:
            model = getattr(self, name)
            model.train()

    # =============================================== SAVE MODELS ======================================================
    def save_networks(self, saveIdx):
        '''
        Save the current registered modules in the /networks folder. Please consider the saveDir has to be configured
        such that the model van be saved.
        :param saveIdx: save index that will be attached to the name

        '''
        assert hasattr(self, 'cfg'), 'No configuration found.'

        for name in self.__modelNames:
            saveName = os.path.join(self.cfg.saveDir, 'networks')

            if not os.path.exists(saveName):
                os.makedirs(saveName)

            saveFile = os.path.join(saveName,'%s_%s.pth' % (saveIdx, name))
            net     = getattr(self, name)


            torch.save(net.cpu().state_dict(), saveFile)
            if torch.cuda.is_available():
                net.cuda()

    # =============================================== LOAD NETWORKS ====================================================
    def load_networks(self, saveIdx):
        '''
        Load the networks from the configured saveDir. Please consider transfering each network to the desired device.
        :param saveIdx: the index to load
        :return:
        '''
        assert hasattr(self, 'cfg'), 'No configuration found'

        for name in self.__modelNames:
            loadName = os.path.join(self.cfg.saveDir, 'networks')
            loadFile = os.path.join(loadName, '%s_%s.pth' % (saveIdx, name))

            model = getattr(self, name)

            print('Load network %s' % loadFile)
            stateDict = torch.load(loadFile)

            if hasattr(stateDict, '_metadata'):
                del stateDict._metadata

            model.load_state_dict(stateDict)

    # =============================================== SAVE OPTMIZERS ===================================================
    def save_optimizers(self, saveIdx):
        assert hasattr(self, 'cfg'), 'No configuration found.'

        for i, name in enumerate(self.__optimizerNames):
            saveName = os.path.join(self.cfg.saveDir, 'optimizers')

            if not os.path.exists(saveName):
                os.makedirs(saveName)

            saveFile = os.path.join(saveName,'%s_%s.pth' % (saveIdx, name))
            torch.save(getattr(self, name).state_dict(), saveFile)

    # =============================================== LOAD OPTMIZERS ===================================================
    def load_optimizers(self, saveIdx):
        assert hasattr(self, 'cfg'), 'No configuration found.'

        for i, name in enumerate(self.__optimizerNames):
            loadName = os.path.join(self.cfg.saveDir, 'optimizers')
            loadFile = os.path.join(loadName, '%s_%s.pth' % (saveIdx, name))

            print('Loading optimizer %s' % loadFile)
            stateDict = torch.load(loadFile)

            if hasattr(stateDict, '_metadata'):
                del stateDict._metadata

            getattr(self, name).load_state_dict(stateDict)

    # =============================================== GET LOSSES =======================================================
    def get_loss_values(self):
        lossDict = OrderedDict()
        for name in self.__lossNames:
            lossDict[name] = float(getattr(self, name+'Value'))

        return lossDict

    # =============================================== GET LOSSES =======================================================
    def get_loss(self):
        lossDict = OrderedDict()
        for name in self.__lossNames:
            lossDict[name] = getattr(self, name)

        return lossDict
    # =============================================== GET OPTIMS =======================================================
    def get_optimizers(self):
        optimizersDict = OrderedDict()
        for name in self.optimizersNames:
            optimizersDict[name] = getattr(self, name)

        return optimizersDict

    # =============================================== GET MODELS =======================================================
    def get_models(self):
        optimizersDict = OrderedDict()
        for name in self.__modelNames:
            optimizersDict[name] = getattr(self, name)

        return optimizersDict

    # =============================================== PRINT NETWORKS ===================================================
    def print_model(self):
        for name in self.__modelNames:
            model = getattr(self, name)
            noParams = 0
            print(model)
            for param in model.parameters():
                noParams += param.numel()

            print('[Network %s] Total no parameters: %.3f M' % (name, noParams / 1e6)) # milions

