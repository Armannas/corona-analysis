# corona-analysis
Some tinkering with COVID-19 data from the European Centre for Disease Prevention and Control.
Plots trendline with chosen days of forecasting/extrapolation. The trendline can be exponential, logistic and or linear. 
You can choose which country to analyze and within what date range.

Predictions can be saved to disk in a Pandas dataframe and compared to the actual value later, to see how well they matched. 
For this, the update_prediction function goes through all previously saved predictions and updates the tables with the latest data.
