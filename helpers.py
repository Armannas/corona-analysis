#
## Author: Arman Nassiri, arman.a.nassiri@gmail.com
#

from scipy.optimize import curve_fit
import numpy as np
import os.path
import definitions
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from functions import func_logit
import urllib.request
import urllib.error

def get_predictions(func, dates, cases, nDays):

    args = dict()
    if func == func_logit:
        args['p0'] = [cases.max(), len(dates) / 2, 0.1, 0],

    args['maxfev'] = 1000000

    x = np.arange(0, len(dates))
    popt, pcov = curve_fit(func, x, cases, **args)

    # 7 day 'forecast'
    datespred = [dates[-1] + timedelta(days=day) for day in range(1, nDays + 1)]
    xpred = np.arange(x[-1] + 1, x[-1] + nDays + 1)
    ypred = func(xpred, *popt)

    # Compute trendline based on fit evaluated for all dates
    dates_all = np.append(dates, datespred)
    x_all = np.append(x, xpred)
    y_all = func(x_all, *popt)

    return datespred, ypred, dates_all, y_all

def load_datasets(country, pop_name, start_date, target):

    ## Download COVID-19 data
    today = str(dt.utcnow().date())
    file_name = "COVID-19-geographic-disbtribution-worldwide-" + today + ".xlsx"
    # Worldwide infections and mortalities from European Center for Disease Control
    url ="https://www.ecdc.europa.eu/sites/default/files/documents/" + file_name

    # If today's data not available, use yesterday's
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError:
        today = str((dt.utcnow() - timedelta(days=1)).date())
        file_name = "COVID-19-geographic-disbtribution-worldwide-" + today + ".xlsx"
        # Worldwide infections and mortalities from European Center for Disease Control
        url = "https://www.ecdc.europa.eu/sites/default/files/documents/" + file_name

    # Same story as with population data
    if os.path.isfile(definitions.ROOT_DIR + '/datasets/' + file_name):
        data = pd.read_excel(definitions.ROOT_DIR + '/datasets/' + file_name)
        print("loaded from disk")
    else:
        # Otherwise download from URL and save to disk
        data = pd.read_excel(url)

        data.to_excel(definitions.ROOT_DIR + '/datasets/' + file_name)
        print("loaded from url")

    data = data.rename(columns={"Cases": "infections", "Deaths": "mortalities"})

    # Choose country to analyze
    data = data[data['Countries and territories'] == country]
    # For NL at least, the numbers are known 1 day earlier than the ECDC claims
    # data['DateRep'] = data['DateRep'] - timedelta(days=1)

    # Sort ascending
    data = data.sort_values('DateRep')

    # Compute cumulative cases from cases per day
    data['infections'] = data['infections'].cumsum()
    data['mortalities'] = data['mortalities'].cumsum()

    # Filter by date
    data = data[data['DateRep'] >= start_date]

    url_pop = "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx"
    file_pop = "population.xls"

    ## Same for population data downloaded from UN
    # Load from disk if available
    if os.path.isfile(definitions.ROOT_DIR + '/datasets/' + file_pop):
        data_pop =  pd.read_excel(definitions.ROOT_DIR + '/datasets/' + file_pop)
    else:
        # Otherwise download from URL and save to disk
        data_pop = pd.read_excel(url_pop, skiprows=16) # Skip first 16 non-data rows

        # Gave up on this, because multiple regions of the same country are included seperately
        # # UN country names are different. Replace by name convention of ECDC
        # for country in pd.unique(data['Countries and territories']):
        #
        #     # Check which element of UN country name matches that of ECDC
        #     # Replace that element with ECDC country name
        #     data_pop.loc[data_pop['Region, subregion, country or area *'].str.contains(country), 'Region, subregion, country or area *'] = country

        data_pop.to_excel(definitions.ROOT_DIR + '/datasets/' + file_pop)

    # Get population for country
    pop = data_pop[(data_pop['Region, subregion, country or area *'] == pop_name)]

    # Choose population in 2020
    pop = np.array(pop['2020'])[0]


    return data, pop