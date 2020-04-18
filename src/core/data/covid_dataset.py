import os

import torch
import numpy  as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data      import Dataset


# =============================================== DATASET ==============================================================
class CovidDataset(Dataset):
    colMapping = {'Date'            : 'date',    'Province_State': 'state',
                    'Country_Region': 'country', 'ConfirmedCases': 'confirmed',
                    'Fatalities'    : 'deaths',  'Id'            : 'id',}

    def __init__(self, root, windowSize = 10, predictSize = 6, batchSize = 32, device = 'cpu', trainEnd = None):
        '''
        Utility class for Kaggle Covid 19 dataset. The output data will be sorted descending.
        :param root: folder root where train.csv is stored
        :param windowSize: window size (total time of prediction)
        :param predictSize: how much of window size will have to be predicted
        :param batchSize: batch size
        :param device: string of where the data will be loaded
        :param trainProc: the percentage of data used for trainin (the rest is for test)
        '''
        self.root        = root
        self.batchSize   = batchSize
        self.predictSize = predictSize
        self.windowSize  = windowSize
        self.trainEnd    = trainEnd
        self.device      = device

        self.rawDf = pd.read_csv(os.path.join(root, 'train.csv'), parse_dates=['Date'])
        self.rawDf   = self.rawDf.rename(columns=CovidDataset.colMapping)
        self.rawDf   = self.rename_states()

        self.rawDf   = self.rawDf[self.rawDf['confirmed'] > 0]

        self.scalers = self.scale_per_country()

        # split in train df/test df
        self.testDf  = self.rawDf[self.rawDf['date'] >= self.trainEnd]
        self.trainDf = self.rawDf[self.rawDf['date'] < self.trainEnd]

        # parse data/label
        self.trainDf = self._parse_countries(self.trainDf)
        # self.testDf  = self._parse_countries(self.testDf)

        self.trainDf = torch.from_numpy(self.trainDf).to(device)
        # self.testDf  = torch.from_numpy(self.testDf).to(device)

    # =============================================== GETITEM =====================================================
    def __getitem__(self, idx):
        '''
        Return batch idx
        :param idx:
        :return:
        '''
        idx   = idx % len(self)
        batch = slice(idx * self.batchSize, (idx + 1) * self.batchSize, 1)

        return self.trainDf[batch, :self.windowSize-self.predictSize,:], \
               self.trainDf[batch, self.windowSize-self.predictSize:,:]

    # =============================================== LEN =====================================================
    def __len__(self):
        return self.trainDf.shape[0]//self.batchSize

    # =============================================== PREPROCESS =====================================================
    def rename_states(self):
        # fill the state field with name of the country (if it is null)
        renameState     = self.rawDf['state'].fillna(0).values
        renameCountries = self.rawDf['country'].values
        renameState[renameState == 0] = renameCountries[renameState == 0]
        self.rawDf['state'] = renameState

        return self.rawDf

    # =============================================== INIT COUNTRY SCALERS =============================================
    def scale_per_country(self):
        scalers = {}
        for state in self.rawDf['state'].unique():
            scaler = StandardScaler()
            d = self.rawDf[self.rawDf['state'] == state][['confirmed', 'deaths']].values

            scaler.fit(d)
            scalers[state] = scaler

        return scalers

    # =============================================== PARSE BATCHES =====================================================
    def _parse_countries(self, df):
        data  = []

        for state in df['state'].   unique():
            # get country data
            d = df[df['state'] == state].sort_values(by='date', ascending=True)
            d = d[['confirmed', 'deaths']]
            d = self.scalers[state].transform(d.values)

            # make each batch (sliding window)
            if d.shape[0] >= self.windowSize:
                for i in range(d.shape[0] - self.windowSize):
                    dat = d[i:i+self.windowSize]
                    data.append(dat)

        # batch, time, size
        data   = np.array(data).astype(np.float32)

        return data
    # =============================================== GET TEST DATA ====================================================
    def get_test_data(self, transform = True):
        resDict = {}

        for state in self.testDf['state'].unique():
            testData = self.testDf[self.testDf['state'] == state][['confirmed', 'deaths']].values

            if transform:
                testData = self.scalers[state].transform(testData)

            testData = torch.from_numpy(testData).to(self.device)

            resDict[state] = testData.float()

        return resDict

if __name__ == '__main__':
    train_up_to = pd.to_datetime('2020-03-25')
    d = CovidDataset('C:\\Users\\beche\\Documents\\GitHub\\kaggle_covid_spread_prediction\\assets' ,trainEnd = train_up_to, batchSize=1)
    print(len(d))
    t = d[1001]
    print(t)
    print(d.get_test_data('Romania'))
    # print(t[0])
