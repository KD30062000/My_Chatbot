#Main Code
#
import cv2
import streamlit as st
import requests
import cv2
import google.generativeai as genai
from streamlit_js_eval import streamlit_js_eval

st.title("MultiModal ChatBot")
# Function to call Google Generative Language API
def call_generative_language_api(message):
    genai.configure(api_key='AIzaSyAD-PGJcGXg6YXJDWdQhGse7VOjy_ENbv8')  # Replace with your actual API key
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(message)
    return response.text


if 'recognized_string' in st.session_state:
    initial_message = st.session_state['recognized_string']
    st.write("Recognized String from Sign Language:", initial_message)
    auto_submit=True
else:
    initial_message = ""
    st.write("No recognized string found. Go to the Sign Language Recognition page and recognize some signs.")
    auto_submit=False
with st.form(key='message_form'):
    message = st.text_area("Enter your message",value=initial_message.lower())
    submit_button = st.form_submit_button(label="Send")

    if submit_button:
        bot_response = call_generative_language_api(message)
        st.text(f"Bot:\n{bot_response}")
if auto_submit:
    streamlit_js_eval(
            js_code="document.querySelector('form').dispatchEvent(new Event('submit', {cancelable: true, bubbles: true}))")

# if 'data' in st.session_state:
#     st.write(st.session_state['data'])
# else:
#     pass
#     st.write("No data found. Go back to Page 1 and enter some data.")






