import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================== PREPROCESS TRAIN =====================================================
def preprocess_data(df):
    '''
    Preprocess the input df from kaggle convid-19 spread competition format.

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

# =============================================== GET SCALER ===========================================================
def get_scaler(allData, target):
    '''
    Fit a standard scaler to target data.
    :param allData: dataframe containing target data
    :param target: fatalities or confirmed
    :return: scaler
    '''
    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    # scale all data
    scaler    = StandardScaler()
    dataScale = allData[colName].values
    dataScale = dataScale.reshape(-1, 1)

    scaler.fit(dataScale)

    return scaler

# ================================================= GET TRAIN DATA =====================================================
def get_train_data(allData, target, trainLimit, winSize, step, scaler = None, shuffle = True):
    '''

    :param allData:     df containing the desired target
    :param target:      can be confirmed or fatalities
    :param trainLimit:  only parse data till this date
    :param winSize:     total window size(observation time + prediction)
    :param step:        step between considering 2 consecutive batches
    :param scaler:      scaler used for normalizing data
    :param shuffle:     shuffle batches
    :return:
    '''
    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    data = allData[allData['Date'] <= trainLimit]

    # get training batches (sliding window)
    batches = []
    for c in data['Province_State'].unique():
        cVals = data[data['Province_State'] == c][colName].values

        for i in range(0, cVals.shape[0] - winSize, step):
            batch = cVals[i:i + winSize].reshape(-1, 1)
            # scale
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
    '''
    :param allData:     df containing the desired target
    :param target:      can be confirmed or fatalities
    :param country:     target country
    :param startFrom:   the first prediction will start from this data
    :param obsSize:     last days until startFrom to be returned for prediction
    :param scaler:      scaler for normalizing the prediction data

    :return: validation data to be fed to the model and all data from strartFrom point
    '''

    colName = 'Fatalities' if target == 'fatalities' else 'ConfirmedCases'

    # select data
    data = allData[allData['Province_State'] == country]

    data = data[data['Date'] >= startFrom - np.timedelta64(obsSize, 'D')][colName].values

    dPred  = data[:obsSize].reshape(-1, 1)
    dLabel = data[obsSize:].reshape(-1, 1)

    if scaler is not None:
        dPred  = scaler.transform(dPred)

    dPred  = torch.Tensor(dPred).unsqueeze(0)
    dLabel = torch.Tensor(dLabel.astype(np.float32))

    return dPred, dLabel
