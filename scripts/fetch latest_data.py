import os
import argparse
import pandas as pd

SAVE_PATH = os.path.join('assets', 'covid_spread.csv')

CONF_LINK = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19' \
               '_time_series/time_series_covid19_confirmed_global.csv'
DEAD_LINK = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19' \
               '_time_series/time_series_covid19_deaths_global.csv'

if __name__ == '__main__':

    check = input('Are you sure you want to update the data? [y/N]')
    if check.lower() != 'y':
        exit(-2)

    # process confirmed cases
    dfConf = pd.read_csv(CONF_LINK)
    dfConf = dfConf.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                         var_name='Date',
                         value_name='ConfirmedCases'
                         ).fillna('').drop(['Lat', 'Long'], axis=1)
    dfConf['Date']           = pd.to_datetime(dfConf['Date'])
    dfConf['ConfirmedCases'] = dfConf['ConfirmedCases'].astype(float)
    dfConf                   = dfConf.sort_values(['Country/Region', 'Province/State', 'Date'])

    # process dead
    dfDead = pd.read_csv(DEAD_LINK)
    dfDead = dfDead.melt(id_vars     = ['Province/State', 'Country/Region', 'Lat', 'Long'],
                          var_name   = 'Date',
                          value_name = 'Fatalities'
                          ).fillna('').drop(['Lat', 'Long'], axis=1)
    dfDead['Date']       = pd.to_datetime(dfDead['Date'])
    dfDead['Fatalities'] = dfDead['Fatalities'].astype(float)

    dfDeadths  = dfDead.sort_values(['Country/Region', 'Province/State', 'Date'])

    df = dfConf.merge(dfDead)
    df = df.rename(columns={'Province/State':'Province_State', 'Country/Region':'Country_Region'})

    df.to_csv(SAVE_PATH, index = False)