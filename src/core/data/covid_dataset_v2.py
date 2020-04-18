import os

import torch
import numpy  as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data      import Dataset


# =============================================== DATASET ==============================================================
from tqdm import tqdm


class CovidDataset(Dataset):
    colMapping = {'Date'            : 'date',    'Province_State': 'state',
                    'Country_Region': 'country', 'ConfirmedCases': 'confirmed',
                    'Fatalities'    : 'deaths',  'Id'            : 'id',}

    def __init__(self, root, windowSize = 10, batchSize = 64, device = 'cpu', trainProc = 0.9):
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
        self.windowSize  = windowSize
        self.trainProc   = trainProc

        self.scaler  = MinMaxScaler([-1, 1])
        self.trainDf = pd.read_csv(os.path.join(root, 'train.csv'), parse_dates=['Date'])
        self.testDf  = pd.read_csv(os.path.join(root, 'test.csv'))
        self.trainDf = self.preprocess_train(self.trainDf)

    # =============================================== GENERATE RELATED =================================================
    def generate_related_countries(self, saveDir, thresh = 1):
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        for state in tqdm(self.trainDf['state'].unique()):
            neighbours = self._get_nearest_sequence(self.trainDf[self.trainDf['confirmed'] >= thresh], state)
            if len(neighbours) > 0:
                neighbours.to_csv(os.path.join(saveDir, state + '.csv'))

        # # augment data
        # self.data = self.parsedData
        # self.data[:,:,0] = np.log10(self.data[:,:,0])
        #
        # # scale data [-1, 1]
        # batch, time = self.data.shape[:2]
        # self.scaler.fit(self.data.reshape(time*batch,-1))
        # self.data = self.scaler.transform(self.data.reshape(batch*time, -1)).reshape(batch, time, -1)
        #
        # # convert to tensor
        # self.data = torch.from_numpy(self.data).float()
        # self.data.to(device)
        #
        # #shuffle
        # order = np.arange(self.data.shape[0])
        # np.random.shuffle(order)
        # self.data[np.arange(self.data.shape[0]), :, :] = self.data[order, :, :]
        #
        # # split
        # trainStep     = int(trainProc * self.data.shape[0])
        # self.testData = self.data[trainStep:, :, :]
        # self.data     = self.data[:trainStep, :, :]

    # =============================================== GETITEM =====================================================
    def __getitem__(self, idx):
        '''
        Return batch idx
        :param idx:
        :return:
        '''
        idx   = idx % len(self)
        batch = slice(idx * self.batchSize, (idx +1) * self.batchSize, 1)

        return self.data[batch, :self.windowSize-self.predictSize,:], \
               self.data[batch, self.windowSize-self.predictSize:,:]

    # =============================================== LEN =====================================================
    def __len__(self):
        return self.data.shape[0]//self.batchSize

    # =============================================== PREPROCESS =====================================================
    def preprocess_train(self, df):

        # fill the state field with name of the country (if it is null)
        df = df.rename(columns=CovidDataset.colMapping)
        renameState     = df['state'].fillna(0).values
        renameCountries = df['country'].values
        renameState[renameState == 0] = renameCountries[renameState == 0]
        df['state'] = renameState

        return df

    # =============================================== PARSE BATCHES =====================================================
    def _get_nearest_sequence(self, df, region):
        resDf = pd.DataFrame(columns=['state', 'deathError', 'confirmedError', 'deathIdx', 'confirmedIdx'])

        # get source region data
        regionDf   = df[df['state'] == region].sort_values(by='date', ascending=True)
        regionConf = regionDf['confirmed'].values
        regionDead = regionDf['deaths'].values

        # check all possible candidates
        for neighbour in df['state'].unique():

            # skip comparing with the same country
            if neighbour == region:
                continue

            # get country candidate
            neighDf = df[df['state'] == neighbour].sort_values(by='date')
            neighConf = neighDf['confirmed'].values
            neighDead = neighDf['deaths'].values

            # get error for confirmed and neighbour
            confErr, confIdx = self._compare_sequence(regionConf, neighConf)
            deadErr, deadIdx = self._compare_sequence(regionDead, neighDead)

            # the candidate will be ignored if it does not have enough data
            if confErr is None or deadErr is None:
                continue

            # append result
            res = {'state':neighbour, 'deathError':deadErr, 'confirmedError':confErr,
                   'deathIdx':deadIdx, 'confirmedIdx':confIdx}
            resDf = resDf.append(res, ignore_index=True)

        resDf = resDf.sort_values(by='confirmedError').reset_index(drop=True)
        return resDf

    # =============================================== COMPARE SEQ ======================================================
    def _compare_sequence(self, source, candidate):
        minError  = np.inf
        minIdx    = -1

        # only check the countries that can influence
        if len(candidate) > len(source):
            noWindows = len(candidate) - len(source)
            windowSize = len(source)

            # sliding window over candidate country
            for i in range(0, noWindows):

                # add 1 for numerical stability
                error = (abs(source + 1 - candidate[i:i+windowSize])).mean()
                # error = error/(source + 1)
                error = error.mean()

                # save the min error
                if error <= minError:
                    minError = error
                    minIdx = i

            return minError, minIdx

        return None, None

    # =============================================== TRANSFORM INVERSE ================================================
    def transform_inverse(self, data):
        '''

        :param data:
        :return:
        '''
        # inverse transform the normalized data
        batch, time = data.shape[:2]
        data = self.scaler.transform(data.reshape(batch * time, -1)).reshape(batch, time, -1)
        data[:, :, 0] = np.pow(10, data[:, :, 0])
        return data


if __name__ == '__main__':
    d = CovidDataset('C:\\Users\\beche\\Documents\\GitHub\\kaggle_covid_spread_prediction\\assets')
    savePath = 'C:\\Users\\beche\\Documents\\GitHub\\kaggle_covid_spread_prediction\\assets\\related'
    d.generate_related_countries(savePath, 2)
    # t = d[10]
    # print(t[0].shape)
    # print(t[0])
