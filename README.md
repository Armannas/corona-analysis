# corona-analysis
Some tinkering with COVID-19 data from the European Centre for Disease Prevention and Control.
Also processes population per country from the United Nations for normalization purposes.

* Dataset contains infections and mortalities
* Plots trendline with arbitrary days of forecasting/extrapolation. 
* The trendline can be exponential, logistic and or linear. 
* You can choose which country to analyze and within what date range.
* Plot absolute values or normalized per 1 million inhabitations
* You can plot multiple countries with different fittings, normalization, plot settings etc. through a dict with params (see code for example)

* Predictions can be saved to disk in a Pandas dataframe and compared to the actual value later, to see how well they matched. 
For this, the update_prediction function goes through all previously saved predictions and updates the tables with the latest data.
