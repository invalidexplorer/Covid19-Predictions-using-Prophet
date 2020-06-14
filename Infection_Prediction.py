.  #!/usr/bin/env python 
# coding: utf-8

# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes 
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation 
import random 
import seaborn as sns 
from fbprophet import Prophet 

cvd = pd.read_csv('D:\EDU\Thapar\Third Year\Sem 6\ML\covid_19_clean_complete.csv' ,parse_dates=True) 

cvd1 = cvd[cvd['Country/Region']=='India'] 
cvd1 = cvd1[['Date','Confirmed']] 
cvd1 = cvd1.rename(columns = {'Date': 'ds', 'Confirmed':'y'} )

m = Prophet() 
m.fit(cvd1) 

future = m.make_future_dataframe(periods = 67) 
forecast = m.predict(future) 

fig = m.plot(forecast, xlabel = 'Date', ylabel = 'Cases') 
fig2 = m.plot_components(forecast) 
forecast 
