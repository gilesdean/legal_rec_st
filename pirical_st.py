import streamlit as st
import numpy as np
import pandas as pd 
from fbprophet import Prophet
import matplotlib.pyplot as plt

df_1 = pd.read_csv('fb_male.csv')
df_2 = pd.read_csv('fb_female.csv')


st.title("Future Offers - Predictive Analysis")

st.write("""

This tool allows predictions to be made on the offers to be made to males and females in future time periods. 

This application is built on the FB Prophet machine learning algorithm to make future predictions. 

Time periods are set to quarterly to smooth the data. 

""")

st.subheader('How many quarters would you like to predict into the future?')

quarters = st.slider("Quarters", 1, 10)

st.subheader('Male Offers Over Time')


m = Prophet(seasonality_mode='additive')
m.fit(df_1)
future = m.make_future_dataframe(periods=quarters,freq='Q')
forecast = m.predict(future)

st.pyplot(m.plot(forecast))

st.subheader('Female Offers Over Time')


m = Prophet(seasonality_mode='additive')
m.fit(df_2)
future = m.make_future_dataframe(periods=quarters,freq='Q')
forecast = m.predict(future)

st.pyplot(m.plot(forecast))