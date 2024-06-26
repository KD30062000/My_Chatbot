
import cv2
import streamlit as st
import requests
import cv2
import google.generativeai as genai

st.title("KD's ChatBot")
# Function to call Google Generative Language API
def call_generative_language_api(message):
    genai.configure(api_key='AIzaSyCqHmnjQAEeVhpBDBBJNwoioTQ0isFiz8M')  # Replace with your actual API key
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(message)
    return response.text

with st.form(key='message_form'):
    message = st.text_area("Enter your message")
    submit_button = st.form_submit_button(label="Send")

    if submit_button:
        bot_response = call_generative_language_api(message)
        st.text(f"Bot:\n{bot_response}")
