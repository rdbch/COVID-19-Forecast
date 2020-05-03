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
                raise NotImplementedError('Init method [%s] not in torch.nn' % weightInit)
            self.weightInit = getattr(torch.nn.init, weightInit)

        self.kwargs = kwargs if kwargs != {} else self.kwargs

        model.apply(self._init_module)

    # =============================================== INIT MODULES =================================
    def _init_module(self, module):
        '''
        Internal function which is applied to every module in a network
        :param module: model on which the function is applied
        '''

        className = module.__class__.__name__

        # init conv and linear layers
        if hasattr(module, 'weight') and ('Conv' in className or 'Linear' in className):
            self.weightInit(module.weight.data, **self.kwargs)

            # init biases
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)

        # init batch norm weight
        # only normal distribution applies
        elif className.find('BatchNorm') != -1:
            torch.nn.init.constant_(module.weight.data, 1.0)
            torch.nn.init.constant_(module.bias.data, 0.0)
