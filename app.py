#Phase 2: Building the Web Application
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('house_model.pkl')

model = load_model()

st.title("🏠 California House Price Prediction App")
st.write("Enter the district details below to predict the median house value.")

# Create input widgets for the 8 features
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in $10k)", value=3.5, step=0.1)
    house_age = st.number_input("House Age (years)", value=25.0, step=1.0)
    ave_rooms = st.number_input("Average Rooms per Household", value=5.0, step=0.1)
    ave_bedrms = st.number_input("Average Bedrooms per Household", value=1.0, step=0.1)

with col2:
    population = st.number_input("Population", value=1400.0, step=10.0)
    ave_occup = st.number_input("Average House Occupancy", value=3.0, step=0.1)
    latitude = st.number_input("Latitude", value=34.0, step=0.1)
    longitude = st.number_input("Longitude", value=-118.0, step=0.1)

# Predict button
if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]],
                              columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    
    prediction = model.predict(input_data)[0]
    # Multiply by 100,000 since target is in $100k units
    final_price = prediction * 100000
    
    st.success(f"Estimated Median House Value: **${final_price:,.2f}**")