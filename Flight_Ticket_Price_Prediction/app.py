# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:39:03 2023

@author: Nithin
"""
import os
import pandas as pd
import pickle
import requests
import streamlit as st
from streamlit_lottie import st_lottie


st.set_page_config(layout="wide", page_title="Predict_Flight_Price", page_icon="✈️")
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
 
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json()

ticket = load_lottieurl("https://raw.githubusercontent.com/nithinganesh1/Deployed_Project/main/Flight_Ticket_Price_Prediction/Animation%20-%201697291145120.json")
feedback = load_lottieurl("https://raw.githubusercontent.com/nithinganesh1/Deployed_Project/main/Flight_Ticket_Price_Prediction/feedback%20Animation.json")

def load_model():
    trained_model_path = "flight_ticket.sav"
    if not os.path.isfile(trained_model_path):
        download_url = "https://github.com/nithinganesh1/Deployed_Project/raw/main/Flight_Ticket_Price_Prediction/flight_ticket.sav"
        download_file(download_url, trained_model_path)
    return pickle.load(open(trained_model_path, 'rb'))

def load_encoder():
    encoder_path = "encoders.pkl"
    if not os.path.isfile(encoder_path):
        download_url = "https://github.com/nithinganesh1/Deployed_Project/raw/main/Flight_Ticket_Price_Prediction/encoders.pkl"
        download_file(download_url, encoder_path)
    return pickle.load(open(encoder_path, 'rb'))


# Loading saved models
trained_model = load_model()
encoders = load_encoder()



#creating a function for prediction
def price_prediction(input_data):
    x_test = pd.DataFrame(columns=['airline', 'source_city', 'departure_time', 'stops', 'arrival_time','destination_city', 'class', 'duration', 'days_left'])
    x_test.loc[0] = input_data
    categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    for col in categorical_cols:
        x_test[col] = encoders[col].transform(x_test[col])
    predict=trained_model.predict(x_test)
    return predict


def main():
    def load_css(url: str):
        r = requests.get(url)
        return r.content

    css = load_css("https://raw.githubusercontent.com/nithinganesh1/Deployed_Project/main/Flight_Ticket_Price_Prediction/style.css")
    st.markdown(f"<style>{css.decode()}</style>", unsafe_allow_html=True)
    
    with st.container():
        st.subheader("hii my name is nithin")
        st.title("Predict Your Flight Ticket Price✈️")
        st.write("This AI-powered website predicts flight prices based on your travel needs")
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with right_column:
            st.header("One simple search.")
            st.write("##")
            st.write(
                """
                Predict your flight ticket price and find the best deals!
                - Enter Your Flight Details.
                - Predict Your Ticket Price.
                - Choose your Best flight 
                - Happy journey
    
                Please review my GitHub profile and provide feedback and suggestions.
                """
            )
            st.write("[ GitHub >](https://github.com/nithinganesh1/Deployed_Project/tree/main/Flight_Ticket_Price_Prediction)")
        with left_column:
            st_lottie(ticket,speed=1,reverse=False,loop=True,quality="low",height=None,width=800,key=None)
            
    # Get the input data from the user
    with st.container():
        st.write("---")
        st.write("##")
        column1, column2,column3 = st.columns((1, 1, 1))
        with column1:
            airline = st.selectbox('Airline', ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo','Air_India'])
            stops = st.selectbox('Stops', ['zero', 'one', 'two_or_more'])
            class_ = st.selectbox('Class', ['Economy', 'Business'])
        with column2:
            source_city = st.selectbox('Source City', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
            arrival_time = st.selectbox('Arrival Time', ['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening', 'Late_Night'])
            duration = st.text_input('Duration',value=1.5)
            if duration != '':
                duration = float(duration)
        with column3:
            departure_time = st.selectbox('Departure Time', ['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night', 'Late_Night'])
            destination_city = st.selectbox('Destination City', ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])
            days_left = st.slider('Days Left', 0, 31,value=3)
            if days_left != '':
                days_left = int(days_left)
        with st.container():
            st.write("##")
            column1, column2 = st.columns((1,2))
            
            with column1:
                st.caption('Click This Button for Prediction')
                if st.button('Predict'):# Make a prediction
                  prediction = price_prediction([airline, source_city, departure_time, stops, arrival_time, destination_city, class_, duration, days_left])
                  with column2:
                      st.markdown(f'<p style="font-size: 45px;"><b>✈ Predicted flight price: <font color="green">{prediction}</font></b></p>', unsafe_allow_html=True)
            st.write("---")
            
    with st.container():
        st.header("Send feedback")
        st.caption("Please fill out this form and click 'Send.' I will receive your message via email.")
        contact_form = """
        <form action="https://formsubmit.co/nithingganesh1@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false"><br><br>
            <input type="text" name="name" placeholder="Your name" required><br><br>
            <input type="email" name="email" placeholder="Your email" required><br><br>
            <textarea name="message" placeholder="Your feedback here" required></textarea><br>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns((1,1))
        with right_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with left_column:
            st_lottie(feedback,speed=1,reverse=False,loop=True,quality="low",height=None,width=400,key=None)


if __name__ == '__main__':
    main()