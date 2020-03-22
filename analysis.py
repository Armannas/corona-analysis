#
## Author: Arman Nassiri, arman.a.nassiri@gmail.com
#

from datetime import timedelta
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import definitions
import os.path
from scipy.optimize import curve_fit
from functions import func_lin, func_logit, func_exp
from functions import setup_dirs
from helpers import load_datasets, get_predictions
from functions import save_prediction, update_predictions

# Setup required directories to save predictions and datasets
setup_dirs()

### Set these values ###
# ----------------------
# The Dutch National Institute for Public Health and the Environment (RIVM)
# has Dutch data a bit earlier than ECDC
rivm_cases = 2994
rivm_deaths = 30

norm_pop_str = ""

params = {
    '1': {
        'country': "Netherlands", # country to analyze
        'start_date': np.datetime64('2020-03-16'), # start date
        'nDays': 5, # Number of days to forecast
        'nDays_off': -16, # offset to allow aligning the data of different countries
        'target': 'mortalities', # variable to analyze (infections or mortalities supported)
        'fit_func': func_exp, # fitting function (func_exp, func_logit and func_lin supported, choose None for no prediction)
        'fit_name': 'exponential', # name of fitting function
        'norm_pop': True, # Normalize per 1 million inhabitants
        'plot_off': 5, # offset location of text for each datapoint
        'plot_marker': "D", # Marker for each datapoint
        'plot_color_known': 'tab:blue', # Color of the known datapoints
        'plot_color_pred': 'darkorange' # Color of the forecasted datapoints
    } ,
     '2': {
         'country': "China",
         'start_date': np.datetime64('2020-02-29'),
         'nDays': 1,
         'nDays_off': 0,
         'target': 'mortalities',
         'fit_func': func_exp,
         'fit_name': 'exponential',
         'norm_pop': True,
         'plot_off': -5,
         'plot_marker': "o",
         'plot_color_known': 'tab:blue',
         'plot_color_pred': 'darkorange'
     }

}
# ----------------------

countries = []
for id in params:

    p = params[id]
    country = p['country']
    countries.append(country)

    # Load COVID-19 and population datasets
    data, pop = load_datasets(country, p['start_date'], p['target'])

    # Update with latest data from RIVM
    # if query_dict['country'] == "Netherlands":
    #     data = data.append({'DateRep': np.datetime64(today), 'Cases': rivm_cases, 'Deaths':rivm_deaths}, ignore_index=True)

    cases = np.array(data[p['target']])

    # Normalize per 1 mil inhabitants
    if p['norm_pop']:
        cases = cases / (pop/1000)
        norm_pop_str = " (per 1 mil inhabitants)"

    # Convert dates to datetime
    dates_pd = pd.to_datetime(np.array(data['DateRep']))
    dates = [d.date() for d in dates_pd]
    date_str = [str(d) for d in dates]

    # Fit a function, choose numerical values for input instead of dates
    if p['fit_func']:
        datespred, ypred, dates_all, y_all = get_predictions(p['fit_func'], dates, cases, p['nDays'])

        # Save today's predictions to disk
        d = dict()
        d['Date'] = datespred
        d['pred'] = ypred.astype(np.int)
        d['acc'] = np.ones(len(ypred)) * np.nan
        d['true'] = np.ones(len(ypred)) * np.nan

        df = save_prediction(dates, datespred, d, p['target'], country, p['fit_name'])

        # Update all predictions with latest data
        d_data = {'Date': dates, 'true': cases}
        df_data = pd.DataFrame(d_data)
        df_data.set_index('Date', inplace=True)

        update_predictions(df_data, p['target'], country, p['fit_name'])

    # Because the outbreak happened at different dates in most countries,
    # it can be easier to read by aligning them. This is done by allowing for an offset in the dates
    dates = [date + timedelta(days=p['nDays_off']) for date in dates]

    if p['fit_func']:
        datespred = [date + timedelta(days=p['nDays_off']) for date in datespred]
    dates_all = [date + timedelta(days=p['nDays_off']) for date in dates_all]

    # Plot known cases
    plt.scatter(dates, cases, c=p['plot_color_known'],marker=p['plot_marker'], label="No. of known " + str(p['target']) + ", " + country + " " + date_str[0] + " - " + date_str[-1], zorder=20)

    if p['fit_func']:
        # Plot fitted trendline
        plt.plot(dates_all, y_all, 'r--', label=f"Estimated trend ({p['fit_name']}, {country})", zorder=1)
        plt.scatter(datespred, ypred, c=p['plot_color_pred'], marker=p['plot_marker'], label = f"Estimated future cases ({p['fit_name']}, {country})", zorder=10)
    else:
        plt.plot(dates, cases, 'r--', label=f"Trend ({p['fit_name']}, {country})", zorder=1)


    plt.xlim(dates_all[0], dates_all[-1])

    # Format the date labels on x axis
    dates_fmt = [date.strftime('%b %-d') for date in dates_all]
    plt.xticks(dates_all, dates_fmt)

    for a,b in zip(dates_all, np.append(cases, ypred)):
        if p['norm_pop']:
            plt.annotate("{:.2f}".format(b), xy=(a, b+p['plot_off']), ha='center')
        else:
            plt.annotate(str(int(b)), xy=(a, b + p['plot_off']), ha='center')


# Print the same country once
countries = np.unique(countries)
plt.title("Number of known Coronavirus " + str(p['target']) + " in " + " and ".join(countries) + ", 7 day forecast")
plt.ylabel("Number of " + str(p['target']) + norm_pop_str)

# Rotate labels for easier reading
plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 45)
plt.legend()
plt.show()

# Open specific prediction
# import pickle
# with open(definitions.ROOT_DIR + '/predictions/' + str(p['target']) + '/' + country + '-' + '17-03-2020.pkl', 'rb') as handle:
#     df = pickle.load(handle)



