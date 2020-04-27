import numpy    as np
import pandas   as pd
import seaborn  as sns

import matplotlib.pyplot as plt

from core.nn.loss import l1_norm_error, rmsle_error

# =============================================== COMPARE SEQUENCE =====================================================
def compare_sequence(source, candidate, errorFunc):
    minError = np.inf
    minIdx = -1

    # only check the countries that can influence
    if len(candidate) > len(source):
        noWindows = len(candidate) - len(source)
        windowSize = len(source)

        # sliding window over candidate country
        for i in range(0, noWindows):

            # compute loss
            error = errorFunc(source, candidate[i:i + windowSize])

            # save the min error
            if error <= minError:
                minError = error
                minIdx = i

        return minError, minIdx

    return None, None

# =============================================== GET NEAREST SEQUENCE =====================================================
def get_nearest_sequence(df, state, alignThreshConf=50, alignThreshDead=10, errorFunc=l1_norm_error):
    resDf = pd.DataFrame(columns=['Province_State', 'deathError', 'confirmedError', 'deathIdx', 'confirmedIdx'])
    confDf = df[df['ConfirmedCases'] > alignThreshConf]
    deadDf = df[df['Fatalities'] > alignThreshDead]

    # get source region data
    regionDfConf = confDf[confDf['Province_State'] == state].sort_values(by='Date', ascending=True)
    regionDfDead = deadDf[deadDf['Province_State'] == state].sort_values(by='Date', ascending=True)

    regionConf = regionDfConf['ConfirmedCases'].values
    regionDead = regionDfDead['Fatalities'].values

    # check all possible candidates
    for neighbour in df['Province_State'].unique():

        # skip comparing with the same country
        if neighbour == state:
            continue

        # get country candidate
        confNeighDf = confDf[confDf['Province_State'] == neighbour].sort_values(by='Date', ascending=True)
        deadNeighDf = deadDf[deadDf['Province_State'] == neighbour].sort_values(by='Date', ascending=True)

        neighConf = confNeighDf['ConfirmedCases'].values
        neighDead = deadNeighDf['Fatalities'].values

        # get error for confirmed and neighbour
        confErr, confIdx = compare_sequence(regionConf, neighConf, errorFunc)
        deadErr, deadIdx = compare_sequence(regionDead, neighDead, errorFunc)

        # the candidate will be ignored if it does not have enough data
        if confErr is None or deadErr is None:
            continue

        # append result
        res = {'Province_State': neighbour, 'deathError': deadErr, 'confirmedError': confErr,
               'deathIdx': deadIdx, 'confirmedIdx': confIdx}

        resDf = resDf.append(res, ignore_index=True)

    return resDf

# =============================================== SHOW CONTRIES =====================================================
def show_country_nn(data, sourceState, alignThreshConf, alignThreshDead, listErrorDf, errorNames):
    SHOW_FIRST = 3  # only show the first top neighbours

    # setup plot figures
    fig, axes = plt.subplots(len(listErrorDf), 2, figsize=(15, len(listErrorDf) * 3), gridspec_kw={'hspace': 0.3})
    axes = axes.flatten()

    fig.suptitle(sourceState.title() + ' - similar growth', fontsize=20)
    colors = sns.color_palette()[:SHOW_FIRST + 1]

    # only keep aligned data
    showDataConf = data[data['ConfirmedCases'] > alignThreshConf].copy()
    showDataDead = data[data['Fatalities'] > alignThreshDead].copy()
    showData = [showDataConf, showDataDead]

    for i, (attr, err) in enumerate(zip(['ConfirmedCases', 'Fatalities'], ['confirmedError', 'deathError'])):
        for j, (error, name) in enumerate(zip(listErrorDf, errorNames)):
            legend = []
            axIdx = j * 2 + i
            tempError = error.sort_values(by=err, ascending=True)

            # of there are less than SHOW_FIRST, on ly disaply what is available
            show = min(SHOW_FIRST, tempError.shape[0])

            for k in range(1, show + 1):

                # plot neighbours
                neighbour = tempError['Province_State'].iloc[k - 1]
                tempShow = showData[i][showData[i]['Province_State'] == neighbour][attr]
                xAxisValues = [z for z in range(tempShow.shape[0])]

                if len(xAxisValues) > 0:
                    legend.append(neighbour)

                sns.lineplot(x=xAxisValues, y=tempShow, color=colors[k], ax=axes[axIdx], linewidth=4.5)

            # plot source country
            tempShow = showData[i][showData[i]['Province_State'] == sourceState][attr]
            xAxisValues = [z for z in range(tempShow.shape[0])]
            sns.lineplot(x=xAxisValues, y=tempShow, color=colors[0], ax=axes[axIdx], linewidth=4.5)

            # final touches to figure
            axes[axIdx].legend(legend + [sourceState])
            axes[axIdx].set_title(name.title() + ' error')
            axes[axIdx].grid(True)

    return axes