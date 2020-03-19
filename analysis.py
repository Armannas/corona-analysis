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

today = str(dt.utcnow().date())
file_name = "COVID-19-geographic-disbtribution-worldwide-" + today + ".xlsx"
# Worldwide infections and mortalities from European Center for Disease Control
url ="https://www.ecdc.europa.eu/sites/default/files/documents/" + file_name

### Set these values ###
# ----------------------
# The Dutch National Institute for Public Health and the Environment (RIVM)
# has Dutch data a bit earlier than ECDC
rivm_cases = 409
rivm_deaths = 18

# Analyze reported infections or mortalities
target = 'mortalities' # mortalities or infections

# Fit logistic function
plot_logistic = False
plot_linear = False

# Length of forecast in days
nDays = 7

source = 'url' # or file

# Filter by country and start date for infections and mortality analysis
query_dict = {'country': 'Netherlands',
              'start_date_inf': np.datetime64('2020-02-26'),
              'start_date_mort': np.datetime64('2020-03-05')}
# ----------------------

# Setup required directories to save predictions and datasets
setup_dirs()

# Load data from file or url
if source == 'url':
    # If file already downloaded, load from disk
    if os.path.isfile(definitions.ROOT_DIR + '/datasets/' + file_name):
        data = pd.read_excel(definitions.ROOT_DIR + '/datasets/' + file_name)
        print("loaded from disk")
    else:
        # Otherwise download from URL and save to disk
        data = pd.read_excel(url)
        data.to_excel(definitions.ROOT_DIR + '/datasets/' + file_name)
        print("loaded from url")

# Load some file directly from disk
else:
    data = pd.read_csv('Corona_NL.csv')


# Choose country to analyze
data = data[data['Countries and territories'] == query_dict['country']]

# For NL at least, the numbers are known 1 day earlier than the ECDC claims
data['DateRep'] = data['DateRep'] - timedelta(days=1)

# Sort ascending
data = data.sort_values('DateRep')

# Update with latest data from RIVM
if query_dict['country'] == "Netherlands":
    data = data.append({'DateRep': np.datetime64(today), 'Cases': rivm_cases, 'Deaths':rivm_deaths}, ignore_index=True)

# Compute cumulative cases from cases per day
if target == 'infections':
    data = data[data['DateRep'] >= query_dict['start_date_inf']]
    cases = np.array(data['Cases'].cumsum())
    plot_offs = 200
    plot_offs_lin = 0
else:
    data = data[data['DateRep'] >= query_dict['start_date_mort']]
    cases = np.array(data['Deaths'].cumsum())
    plot_offs = 10
    plot_offs_lin = 10

# Convert dates to datetime
dates_pd = pd.to_datetime(np.array(data['DateRep']))
dates = [d.date() for d in dates_pd]

date_str = [str(d) for d in dates]
# date_str = [date.date()]
# dates = [dt.strptime(date, '%m/%d/%Y') for date in date_str]

# Fit a function, choose numerical values for input instead of dates
x = np.arange(0, len(dates))
popt_exp, pcov_exp = curve_fit(func_exp, x, cases)
popt_logit, pcov_logit = curve_fit(func_logit, x, cases,p0=[cases.max(), len(dates)/2,0.1, 0], maxfev=1000000)
popt_lin, pcov_lin = curve_fit(func_lin, x, cases)

# 7 day 'forecast'
datespred = [dates[-1] + timedelta(days=day) for day in range(1, nDays + 1)]
xpred = np.arange(x[-1] + 1, x[-1] + nDays + 1)
ypred_exp = func_exp(xpred, *popt_exp)
ypred_logit = func_logit(xpred, *popt_logit)
ypred_lin = func_lin(xpred, *popt_lin)

# Draw trendline based on fit evaluated for all dates
dates_all = np.append(dates, datespred)
x_all = np.append(x, xpred)
y_all_exp = func_exp(x_all, *popt_exp)
y_all_logit = func_logit(x_all, *popt_logit)
y_all_lin = func_lin(x_all, *popt_lin)

# Plot known cases
plt.scatter(dates, cases,label="No. of known " + str(target) + " " + date_str[0] + " - " + date_str[-1], zorder=20)

# Plot fitted trendline
plt.plot(dates_all, y_all_exp, 'r--', label="Estimated trend (exponential)", zorder=1)
plt.scatter(datespred, ypred_exp, label = "Estimated future cases (exponential)", zorder=10)

if plot_logistic:
    plt.plot(dates_all, y_all_logit, 'g--', label="Estimated trend (logistic)", zorder=1)
    plt.scatter(datespred, ypred_logit, label = "Estimated future cases (logistic)", zorder=10)

if plot_linear:
    plt.plot(dates_all, y_all_lin, 'r--', label="Estimated trend (linear)", zorder=1)
    plt.scatter(datespred, ypred_lin, label="Estimated future cases (linear)", zorder=10)

plt.title("Number of known Coronavirus " + str(target) + " in " + query_dict['country'] + " and 7 day forecast")
plt.ylabel("Number of " + str(target))
plt.xlim(dates_all[0], dates_all[-1])

# Format the date labels on x axis
dates_fmt = [date.strftime('%b %-d') for date in dates_all]
plt.xticks(dates_all, dates_fmt)
for a,b in zip(dates_all, np.append(cases, ypred_exp)):
    plt.annotate(str(int(b)), xy=(a-timedelta(hours=0), b+plot_offs), ha='center')

if plot_logistic:
    for a,b in zip(datespred, ypred_logit):
        plt.annotate(str(int(b)), xy=(a, b-plot_offs-plot_offs_lin), ha='center')

if plot_linear:
    for a,b in zip(datespred, ypred_lin):
        plt.annotate(str(int(b)), xy=(a, b-plot_offs-20), ha='center')

# Rotate labels for easier reading
plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 45)
plt.legend()
plt.show()

#%% Save today's predictions to disk
from functions import save_prediction, update_predictions

d = {'Date': datespred,
     'pred_exp': ypred_exp.astype(np.int),
     'pred_logit': ypred_logit.astype(np.int),
     'true': np.ones(len(ypred_exp)) * np.nan,
     'acc_exp': np.ones(len(ypred_exp)) * np.nan,
     'acc_logit': np.ones(len(ypred_logit)) * np.nan}
if plot_linear:
    d['pred_lin'] = ypred_lin.astype(np.int)
    d['acc_lin'] = np.ones(len(ypred_lin)) * np.nan

df = save_prediction(dates, datespred, d, target)


#%% Update all predictions with latest data
d_data = {'Date': dates, 'true': cases}
df_data = pd.DataFrame(d_data)
df_data.set_index('Date', inplace=True)

update_predictions(df_data, target)

#%% Open specific prediction
import pickle
with open(definitions.ROOT_DIR + '/predictions/' + str(target) + '/' + '17-03-2020.pkl', 'rb') as handle:
    df = pickle.load(handle)



