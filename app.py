from src.logger import logging
from src.exception import CustomException
from src.pipelines.prediction_pipeline import Predicion_pipe
import pandas as pd 
import numpy as np
from src.utils import load_object

import streamlit as st 
left_col, right_col = st.columns([2, 8])

# Display the image in the left column
with left_col:
    st.image('1701097834232.jpg', width=300)  # Adjust width as needed

# Display the title in the right column
with right_col:
    st.title("Ensuring Optimal Solutions for Cement Strength Prediction by Piyush Dhondiyal")

st.title('Cement strength predictor')

cement = st.number_input('Enter cement volume in cubic metres')
Blast_Furnace_Slag = st.number_input('Enter Blast Funace Slag in cubic metre')
Fly_Ash = st.number_input('Enter Fly Ash in cubic metre')
Water = st.number_input('Enter water in cubic metre')
Superplasticizer = st.number_input('Enter Superplasticizer in cubic metre')
Coarse_Aggregate = st.number_input('Enter Coarse Aggregate in cubic metre')
Fine_Aggregate = st.number_input('Enter Fine Aggregate in cubic metre')
Age = st.number_input('Enter the age')

data = {}
if st.button('predict strength'):
    data = {
    'Cement':cement,
    'Blast Furnace Slag':Blast_Furnace_Slag,
    'Fly Ash':Fly_Ash,
    'Water':Water,
    'Superplasticizer':Superplasticizer,
    'Coarse Aggregate':Coarse_Aggregate,
    'Fine Aggregate':Fine_Aggregate,
    'Age':Age}
    making_prediction = Predicion_pipe()
    result = making_prediction.making_prediction(data)
    

    st.title(result)


