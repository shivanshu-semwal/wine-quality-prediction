import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

# loading model

# loading scaler
scale_X = np.load('model/scale.npy')
means = scale_X[0]
vars = scale_X[1]
stds = vars ** 0.5

high_eg = [7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0]
medium_eg = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
low_eg = [5.7, 1.13, 0.09, 1.5, 0.172, 7.0, 19.0, 0.994, 3.5, 0.48, 9.8]

if __name__ == "__main__":
    st.title('Wine Quality Prediction')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity: ", 0.0, 10.0, step=0.1)
        volatile_acidity = st.number_input(
            "Volatile acidity: ", 0.0, 1.0, step=0.1)
        citric_acid = st.number_input("Citric Acid: ", 0.0, 1.0, step=0.1)
    with col2:
        residual_sugar = st.number_input(
            "Residual Sugar: ", 0.0, 5.0, step=0.1)
        chlorides = st.number_input("Chlorides: ", 0.0, 1.0, step=0.1)
        free_sulfur_dioxide = st.number_input(
            "Free sulfur dioxide: ", 0.0, 20.0, step=0.1)

    with col3:
        total_sulfur_dioxide = st.number_input(
            "Total sulfur dioxide: ", 0.0, 100.0, step=1.0)
        density = st.number_input("Density: ", 0.0, 2.0, step=0.1)
        pH = st.number_input("pH: ", 2.0, 5.0, step=0.1)

    with col4:
        sulphates = st.number_input("Sulphates: ", 0.0, 1.0, step=0.1)
        alcohol = st.number_input("Alcohol: ", 0.0, 10.0, step=0.1)

    val = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
           total_sulfur_dioxide, density, pH, sulphates, alcohol]
    val = np.array(val)
    val = (val - means)/stds

    model = pickle.load(open('model/model.sav', 'rb'))
    ans = model.predict(np.expand_dims(val, axis=0))[0]

    if ans == "HIGH":
        st.markdown('> *High* Quality Wine')
    elif ans == "LOW":
        st.markdown('> *Low* Quality Wine')
    else:
        st.markdown('> *Medium* Quality Wine')