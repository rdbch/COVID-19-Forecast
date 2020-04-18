import os
import yaml
import inspect
import functools

from yaml import Dumper, Loader

# =============================================== CONFIGURABLE =========================================================
class Configurable:
    '''A utility class for creating configurable objects.
    A class can have
        - internal configuration
        - external configuration

    '''
    # =============================================== VALIDATE INTERNAL ================================================
    def _validate_external_config(self):
        '''
        Utility function for validating the external configuration. It is supposed that the external configuration is
        already set in self.

        Example:
            assert self.param > 0, "Param not ok"

        '''
        pass

    # =============================================== VALIDATE EXTERNAL ================================================
    def _validate_internal_config(self, kwargs):
        '''
         Utility function for validating the internal configuration. It is supposed that the internal configuration is
         set. Its attribute can be accesed via the kwargs arguments.

         Example:
            assert kwargs['paramName'] > 0, 'Param not ok.Param is bad'

        :param kwargs: the internal configuration dictionary
        '''
        pass

    # =============================================== BUILD ============================================================
    def build(self):
        raise NotImplementedError("Please initialize all custom objects here")

    # =============================================== EXTERNAL CONFIG ==================================================
    def build_external_config(self, validate = True):
        '''
        External configuration are field that are passed via the __init__ method in the subclass as named arguments
        (both positional and keywords). They also have to be set in the self object having the same name as in the
        contrustor signature.

        Example:
            def __init__(self, a b, d= 3, *args, **kwargs)
                self.a = a
                self.anotherNameForB = b
                self.d = d

            Only <a> and <d> will be considered external configuration parameters, as they are named parameters, and
            they are both set in the method signature and set into the object

        As a side effect, calling this function will create a list of external configuration will keep the name of these
        parametrs. This list will be called when saving the configuration.
        :return:
        '''
        assert not hasattr(self, 'externalConfigAttrs'), "Please do not use [externalConfigAttrs] field."

        externalConfigAttrs = []
        initSig = inspect.signature(self.__init__)

        for param in initSig.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                if hasattr(self, param.name):
                    externalConfigAttrs.append(param.name)

        if validate:
            self._validate_external_config()

        setattr(self, '__externalConfigAttrs', externalConfigAttrs)

    # =============================================== INTERNAL CONFIG ==================================================
    @staticmethod
    def internal_config(validate=True):

        def actual_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = func(*args, **kwargs)

                # args[0] refferes to self

                for key in res.keys():
                    res[key] = kwargs.get(key, res[key])
                    setattr(args[0], key, res[key])

                if validate:
                    args[0]._validate_internal_config(res)

                setattr(args[0], '__internalConfigAttrs', [str(key) for key in res.keys()])

                return res
            return wrapper
        return actual_decorator

    # =============================================== SAVE CONFIG ======================================================
    def save_config(self, path, internal = True, external = True):
        '''
        Utility method for saving the configuration dictionaries.
        :param internal: path for saving the internal config
        :param external: path for saving the external config

        '''

        saveDict = {}

        if internal:
            assert hasattr(self, '__internalConfigAttrs'), 'Can not find internal config'
            saveDict['internal'] = self.get_config_dict('internal', False, True)

        if external:
            assert hasattr(self, '__externalConfigAttrs'), 'Can not find external config'
            saveDict['external'] = self.get_config_dict('external', False, True)

        # get the directory and create it if it does not exists
        pathDir = os.path.split(path)[0]

        if not os.path.exists(pathDir) and pathDir is not '':
            os.makedirs(pathDir)

        # save the configuration
        with open(path, 'w') as f:
            yaml.dump(saveDict, f, Dumper)

    # =============================================== LOAD CONFIG ======================================================
    def load_config(self, path, internal = True, external = True, loadExisting = True, validate = True):
        '''
        Utility function for loading a desired configuration.
        :param internal: path for loading the internal config
        :param external: path for loading the external config
        :param loadExisting: use the existing keys in list or use the loaded keys
        :return:
        '''

        assert os.path.exists(path), 'The config file could not be found: %s' % path

        with open(path, 'r') as f:
            myDict = yaml.load(f, Loader)

        if internal: assert 'internal' in myDict, 'Could not find internal config into dict'
        if external: assert 'external' in myDict, 'Could not find external config into dict'

        # only load the keys existing in the actual configuration
        if loadExisting:
            if internal:
                assert hasattr(self, '__internalConfigAttrs'), 'The object does not have internal config set.'

                if validate :
                    self._validate_internal_config(myDict['internal'])

                for attr in getattr(self, '__internalConfigAttrs'):
                    if attr in myDict['internal']:
                        setattr(self, attr, myDict['internal'][attr])
                    else:
                        raise ValueError('Could not load attr %s from dict with keys: %s' % \
                                         (attr, ['%s , ' % str(key) for key in myDict['internal'].keys()]))

            if external:
                assert hasattr(self, '__externalConfigAttrs'), 'The object does not have external config set.'

                for attr in getattr(self, '__externalConfigAttrs'):
                    if attr in myDict['external']:
                        setattr(self, attr, myDict['external'][attr])
                    else:
                        print('Warning: ignored ',attr ,'key')
                        # raise ValueError('Could not load attr %s from dict with keys: %s' % \
                        #                  (attr, ['%s , ' % str(key) for key in myDict['external'].keys()]))

                if validate :
                    self._validate_external_config()

        # load the all the keys that are in the configuration value
        else:
            if internal:
                if validate:
                    self._validate_internal_config(myDict)

                setattr(self, '__internalConfigAttrs', [str(key) for key in myDict.keys()])
                for key in myDict['internal'].keys():
                    setattr(self, key, myDict[key])

            if external:
                setattr(self, '__externalConfigAttrs', [str(key) for key in myDict.keys()])
                for key in myDict['external'].keys():
                    setattr(self, key, myDict[key])

                if validate:
                    self._validate_external_config()

        return self

    # =============================================== GET CONFIG DICT ==================================================
    def get_config_dict(self, name, returnMissing = False, includeMisssing = True):
        '''
        Build and get the configuration dictionary.
        :param name: [internal/external]
        :param returnMissing: flag for returning the missing attributes
        :return a dict for the specified configuration
        '''

        # validate parameters
        cfgName = '__' + name + 'ConfigAttrs'
        assert name in ['internal', 'external'], 'Please choose from [internal/external]. Currently [%s]' % name
        assert  hasattr(self,cfgName),            'The object does not have this config set: %s' % cfgName

        configDict   = {}
        missingAttrs = []

        for attr in getattr(self, '__'+name+'ConfigAttrs'):
            # if the object does not have the attribute, store it
            if hasattr(self, attr):
                configDict[attr] = self.__dict__[attr]
            else:
                if includeMisssing:
                    configDict[attr] = None
                missingAttrs.append(attr)

        # return the desired parameters
        if returnMissing:
            return configDict, missingAttrs

        return configDict

    # =============================================== PRINT CONFIG =====================================================
    def print_config(self):
        print('############# CONFIG ##############')
        print('------------- INTERNAL ------------')
        print(yaml.dump(self.get_config_dict('internal'), default_flow_style=False))
        print('------------- EXTERNAL ------------')
        print(yaml.dump(self.get_config_dict('external'), default_flow_style=False))


# =============================================== TEST =================================================================
if __name__ == '__main__':
    class A(Configurable):
        def __init__(self, a, b, c, d=5, **kwargs):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.build_external_config()

            self.mehe = 123
            self.build_internal_config(**kwargs)

        @Configurable.internal_config()
        def build_internal_config(self, **kwargs):
            return {'aha' : 3, 'heeh':'sad'}

        def _validate_internal_config(self, kwargs):
            assert kwargs['aha'] > 2, 'tztz'

    a = A(1,2,3,4, hohoho=5, heeh=11212)

    a.save_config('path.yaml')
    a.load_config('path.yaml')
    a.print_config()