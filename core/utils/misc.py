import os
import torch

import seaborn           as sns
import matplotlib.pyplot as plt

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

# =============================================== VISUALIZE PREDICTION =================================================
def visualize_predictions(model, countryName, scaler, ):
    model.eval()

    # get figure
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    fig.suptitle(countryName + ' prediction')

    # prepera data for prediction
    d = confirmedTrain[confirmedTrain['Province_State'] == countryName]
    dPred = confScaler.transform(d['ConfirmedCases'].values.reshape(-1, 1))
    dPred = torch.Tensor(dPred).unsqueeze(0).to(DEVICE)

    # make prediction
    pred = model(dPred[:, -OBS_SIZE:], future=50).cpu().detach().numpy()
    pred = confScaler.inverse_transform(pred[0])

    # plot prediction
    predDate = pd.date_range(start=d['Date'].values[-OBS_SIZE], periods=pred.shape[0])
    sns.lineplot(y=pred, x=predDate, ax=ax)

    # plot train data
    dPred = confScaler.inverse_transform(dPred[0].cpu())
    sns.lineplot(y=dPred[:, 0], x=d['Date'], ax=ax)

    # plot validation
    valData = confirmedVal[confirmedVal['Province_State'] == COUNTRY]['ConfirmedCases']
    valDate = confirmedVal[confirmedVal['Province_State'] == COUNTRY]['Date']
    sns.lineplot(y=valData, x=valDate, ax=ax);

    ax.legend(['Train', 'Pred', 'Validation'])
    ax.grid(True)