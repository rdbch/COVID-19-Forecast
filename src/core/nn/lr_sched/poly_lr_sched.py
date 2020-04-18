import torch
from torch.optim.lr_scheduler import _LRScheduler

# ================================================== POLY LR SCHED =====================================================
class PolyLrSched(_LRScheduler):
    def __init__(self, optimizer, stepSize, iterMax, powerDecay=0.9, warmupIters=500, warmupFactor = 0.33, lastEpoch=-1):
        '''
        Polynomial decay learning rate scheduler with warm-up phase capability. For the first <warmUp> iters the lwarning period
        rate will be multiplies with the <warumFactor> (which is usually between [0, 1]. After that, the decay will
        occur after the following formula:

            lr * (1 -iter/maxIter) ** power

        :param optimizer: the optimizer to use
        :param stepSize: the step size within the learning rate will be updated
        :param iterMax: the last iterarion/epoch number
        :param powerDecay: the decay power (0.9)
        :param warmupIters: the number of iters (no steps) up until you apply the warm-up factor
        :param warmupFactor: an number ti be multiplied with the leaning rate during the warm-up period
        :param lastEpoch:
        '''
        self.stepSize = stepSize
        self.iterMax  = iterMax
        self.power    = powerDecay

        self.warmupFactor = warmupFactor
        self.warmupIters  = warmupIters

        self.lastLr = []
        super(PolyLrSched, self).__init__(optimizer, lastEpoch)

    # ================================================== DECAY =========================================================
    def decay(self, lr):
        '''
        The decay function.
        :param lr: the lr on which to apply the decay
        :return:
        '''
        return lr * (1 - self.last_epoch / self.iterMax) ** self.power

    # ================================================== GET LR ========================================================
    def get_lr(self):
        # when a step occurs update the learning rate
        if self.last_epoch % self.stepSize == 0:

            # warmup phase
            if self.last_epoch < self.warmupIters:
                self.lastLr = [self.decay(lr) * self.warmupFactor for lr in self.base_lrs]
                return self.lastLr

            # after warmup, normal decay
            self.lastLr =  [self.decay(lr) for lr in self.base_lrs]
            return self.lastLr

        # keep the same lr as the last one
        return self.lastLr

# ================================================== TEST ==============================================================
if __name__ == '__main__':
    from torch.optim import Adam
    import matplotlib.pyplot as plt

    adam = Adam([torch.randn(2)], lr=0.007)

    lrSched = PolyLrSched(
        adam,
        stepSize = 200,
        iterMax = 18500,
        powerDecay = 0.9,
        warmupIters = 2000,
        warmupFactor = 0.33,
        lastEpoch=-1)
    idxs = []
    vals = []
    for i in range(100):
        for batch in range(185):
            adam.step()
            lrSched.step()

        idxs.append(i)
        vals.append(lrSched.get_lr())

    plt.plot(idxs, vals)
    plt.show()
