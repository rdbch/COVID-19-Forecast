import sys
import time
import visdom
import numpy as np

from subprocess import Popen, PIPE
# ================================================== VISUALIZER ========================================================
class VisdomClient:
    def __init__(self, hostIp = '127.0.0.1', port=2000, env='Land'):
        self.hostIp = hostIp
        self.port   = port
        self.env    = env
        self.vis    = visdom.Visdom(server=self.hostIp, port=self.port, env=self.env)

    # ================================================== CREATE CONN ===================================================
    def start_server(self):
        cmd = sys.executable + ' -m visdom.server -p %d ' % self.port
        Popen(cmd)
        time.sleep(10)


    # ================================================== PLOT PARAM ====================================================
    def plot_param(self, stepNo, paramDict, winName):
        self.vis.line(X = np.array([stepNo]),
                      Y = np.column_stack([np.array(val) for key, val in paramDict.items()]),
                      opts = {
                        'title'  : winName.title() + ' Monitor',
                        'legend' : list(paramDict.keys()),
                        'xlabel' : 'Steps',
                        'ylabel' :  winName},
                    win  = winName + ' Scalar Window',
                    update = 'append')

    # ================================================== PLOT PARAMS ===================================================
    def plot_params(self, stepNo, paramDict):
        for key, value in paramDict.items():
            self.vis.line(X = np.array([stepNo]),
                          Y = np.array([value]),
                          opts = {
                            'title'  : str(key).title() + ' Monitor',
                            'legend' : list(metrics.keys()),
                            'xlabel' : 'Steps'},
                        win  = str(key).title() + ' Scalar Window',
                        update = 'append')

    # =============================================== SAVE =============================================================
    def save(self):
        '''Save the current env in the local cache'''
        self.vis.save([self.env])

# ================================================== TEST ==============================================================
if __name__ == '__main__':
    a = Visualizer()
    a.build()
    time.sleep(10)