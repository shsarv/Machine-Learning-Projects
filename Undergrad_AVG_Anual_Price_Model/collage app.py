# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:42:07 2023

@author: Nithin
"""

import os
import pandas as pd
import pickle
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="collage_avg_price", page_icon="🎓")

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

def load_model():
    trained_model_path = "collage_price_model.sav"
    if not os.path.isfile(trained_model_path):
        download_url = "https://github.com/nithinganesh1/Deployed_Project/raw/main/Undergrad_AVG_Anual_Price_Model/collage_price_model.sav"
        download_file(download_url, trained_model_path)
    return pickle.load(open(trained_model_path, 'rb'))

def load_transformer():
    transformer_path = "transformer.pkl"
    if not os.path.isfile(transformer_path):
        download_url = "https://github.com/nithinganesh1/Deployed_Project/raw/main/Undergrad_AVG_Anual_Price_Model/transformer.pkl"
        download_file(download_url, transformer_path)
    return pickle.load(open(transformer_path, 'rb'))

def load_scaler():
    scaler_path = "sc.pkl"
    if not os.path.isfile(scaler_path):
        download_url = "https://github.com/nithinganesh1/Deployed_Project/raw/main/Undergrad_AVG_Anual_Price_Model/sc.pkl"
        download_file(download_url, scaler_path)
    return pickle.load(open(scaler_path, 'rb'))

# Loading saved models
trained_model = load_model()
transformer = load_transformer()
scaler = load_scaler()


#creating a function for prediction
def price_prediction(input_data):
    x_test = pd.DataFrame(columns=['Year', 'State', 'Type', 'Length', 'Expense'])
    x_test.loc[0] = input_data
    x_test=transformer.transform(x_test)
    x_test=scaler.transform(x_test)

    predict=trained_model.predict(x_test)
    return f'Expected Price for Average undergraduate tuition and fees and room and board rates is = {predict}'



def main():
    
    #giving a Title
    st.title("Avg Cost Prediction of Undergrad Colleges")
    st.image("https://github.com/nithinganesh1/Deployed_Project/raw/main/Undergrad_AVG_Anual_Price_Model/chartimage.png", width=800)
    
    #data description
    st.markdown("""Compiled from the National Center of Education Statistics Annual Digest. Specifically,
                Table 330.20: Average undergraduate tuition and fees and room and board rates charged for
                full-time students in degree-granting postsecondary institutions,
                by control and level of institution and state or jurisdiction.
                ###Check out!
                - data [nces330_20.csv](https://github.com/nithinganesh1/Deployed_Project/blob/main/Undergrad_AVG_Anual_Price_Model/nces330_20.csv)
                - GitHub [Undergrad_AVG_Anual_Price_Model](https://github.com/nithinganesh1/Deployed_Project/tree/main/Undergrad_AVG_Anual_Price_Model)
                - My_GitHub : [Nithin_Ganesh1](https://github.com/nithinganesh1)""")
    
    #getting the input data from the user
    Year = st.text_input('Year of Joining')
    if Year != '':
        Year = int(Year)
    state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
       'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
       'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
       'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
       'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
       'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
       'New Jersey', 'New Mexico', 'New York', 'North Carolina',
       'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
       'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
       'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
       'West Virginia', 'Wisconsin', 'Wyoming']
    State = st.selectbox('Select Searching State',options = state_list)
    
    st.markdown('<h2 style="color: darkblue; font-family: \'Comic Sans MS\', cursive, sans-serif;">Type Plot</h2>', unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/nithinganesh1/Deployed_Project/main/Undergrad_AVG_Anual_Price_Model/Type.png",width=500)
    type_list = ['Private', 'Public In-State', 'Public Out-of-State']
    Type = st.selectbox('Type of studie',options=type_list)
    
    st.markdown('<h2 style="color: darkblue; font-family: \'Comic Sans MS\', cursive, sans-serif;">Duration Plot </h2>', unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/nithinganesh1/Deployed_Project/main/Undergrad_AVG_Anual_Price_Model/Length.png",width=500)
    Length = st.text_input('Duration of the course in Year')
    if Length != '':
        Length = int(Length)
        
    st.markdown('<h2 style="color: darkblue; font-family: \'Comic Sans MS\', cursive, sans-serif;">Expense Plot</h2>', unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/nithinganesh1/Deployed_Project/main/Undergrad_AVG_Anual_Price_Model/Expense.png", width=500)
    expense_list = ['Fees/Tuition', 'Room/Board']
    Expense = st.selectbox('Type of Expense',options=expense_list)
    
    #code for prediction
    diagnosis = ''
    
    #creating a botton for prediction
    
    if st.button('Price_Prediction'):
        diagnosis = price_prediction([Year,State,Type,Length,Expense])
        
    st.markdown(f'<h1 style="color: green;">{diagnosis}</h1>', unsafe_allow_html=True)

    

if __name__ == '__main__':
    main()




        
    
    
    
    
    
    
    