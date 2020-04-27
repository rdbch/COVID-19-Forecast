import torch
import numpy as np
import pandas as pd
# =============================================== PREPROCESS TRAIN =====================================================
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    '''
    Preprocess the input dataframe from the Kaggles's COVID-19 spread prediction.

    Fill empty states with their country names (easier future handling).
    :param df: raw dataframe
    :return: dataframe
    '''
    # fill the state field with name of the country (if it is null)
    renameState     = df['Province_State'].fillna(0).values
    renameCountries = df['Country_Region'].values
    renameState[renameState == 0] = renameCountries[renameState == 0]
    df['Province_State'] = renameState

    return df

# =============================================== GET TARGET DATA =====================================================
def get_target_data(allData, errorData, errorThresh, country, target):
    errName = 'deathError' if target == 'fatalities' else 'confirmedError'
    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    errorDf = errorData.sort_values(by=errName)

    topConfCountries = errorDf.sort_values(by=errName)
    topConfCountries = topConfCountries[topConfCountries[errName] < errorThresh]
    topConfCountries = topConfCountries['Province_State'].values.tolist()
    topConfCountries.append(country)

    targetCols = ['Date', 'Province_State', colName]
    data = pd.DataFrame(columns=targetCols)

    for country in topConfCountries:
        countryData = allData[allData['Province_State'] == country][targetCols]
        countryData = countryData[countryData[colName] > 0]
        data = pd.concat([data, countryData])

    return data

def get_scaler(allData, target):
    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    # scale all data
    scaler = StandardScaler()
    dataScale = allData[colName].values
    dataScale = dataScale.reshape(-1, 1)
    scaler.fit(dataScale)

    return scaler

# ================================================= GET TRAIN DATA =====================================================
def get_train_data(allData, target, trainLimit, winSize, step, scaler = None, shuffle = True):
    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    data = allData[allData['Date'] <= trainLimit]

    batches = []
    for c in data['Province_State'].unique():
        cVals = data[data['Province_State'] == c][colName].values
        for i in range(0, cVals.shape[0] - winSize, step):
            batch = cVals[i:i + winSize].reshape(-1, 1)
            if scaler is not None:
                batch = scaler.transform(batch)
            batches.append(batch)


    batches = torch.Tensor(batches).float()

    # shuffle input data
    if shuffle:
        order = np.array(range(batches.shape[0]))
        np.random.shuffle(order)
        batches[np.array(range(batches.shape[0]))] = batches[order]

    return batches

# =============================================== GET VAL DATA =========================================================
def get_val_data(allData, target, country, startFrom, obsSize, scaler = None):
    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    # prepera data for prediction
    d     = allData[allData['Province_State'] == country]
    dPred = scaler.transform(d['ConfirmedCases'].values.reshape(-1, 1))

    if scaler is not None:
        dPred = scaler.transform(dPred)

    dPred = torch.Tensor(dPred).unsqueeze(0)

    return dPred

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