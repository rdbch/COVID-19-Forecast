import numpy as np

# =============================================== L1 NORM =====================================================
def l1_norm_error(source, candidate):

    error = (abs(source - candidate))
    source[source == 0] = 1e-30  # add for numerical stability
    error = error / source       # compute the percentage
    error = error.mean()
    return error

# =============================================== RMSLE  =====================================================
def rmsle_error(source, candidate):
    candidate += 1e-30
    error = np.log10((source + 1) / (candidate + 1))
    error = error * error
    error = error.mean()
    error = np.sqrt(error)

    return error