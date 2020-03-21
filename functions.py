#
## Author: Arman Nassiri, arman.a.nassiri@gmail.com
#

import pandas as pd
import pickle
import numpy as np
import definitions
import os.path
def func_exp(x, a, b, c):
    return a * np.exp(b * x) + c
def func_logit(x,A,x0,k,off):
    return A / (1 + np.exp(-k*(x-x0)))+off

def func_lin(x, a, b):
    return a*x + b

def setup_dirs():
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    if not os.path.exists("predictions"):
        os.mkdir("predictions")
        os.mkdir("predictions/infections")
        os.mkdir("predictions/mortalities")

def save_prediction(dates_dt, datespred, d, target, country, fit_name):


    df = pd.DataFrame(d)
    df.set_index('Date', inplace=True)

    if not os.path.exists('predictions/' + target + '/' + country):
        os.mkdir('predictions/' + target + '/' + country)
    with open(definitions.ROOT_DIR + '/predictions/' + target + '/' + country + '/' + dates_dt[-1].strftime('%d-%m-%Y') + '_' + fit_name + '.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return df


import os
def update_predictions(df_data, target, country, fit_name):
    # Get predictions for every day
    for file in os.listdir(definitions.ROOT_DIR + '/predictions/' + target + '/' + country):
        with open(definitions.ROOT_DIR + '/predictions/' + target +'/' + country + '/' + file, 'rb') as handle:
            df = pickle.load(handle)

    # Update every prediction with latest data
        df.update(df_data, overwrite=True)

    # Compute prediction accuracy if true value available
        df['acc'] = 100 * (df['pred'] - df['true']) / df['true']

        with open(definitions.ROOT_DIR + '/predictions/' + target + '/'+ country + '/' + file, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
