import streamlit as st 
import numpy as np 
import pickle
import base64
def set_background_image(image_file):
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    background_style = f"""
    <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        div, h1, h2, h3, h4, h5, h6, p, span, label {{
            color: black !important;  /* Force all text to be black */
            font-weight: bold !important;  /* Make all text bold */
        }}
        .stButton > button {{
            color: black !important;  /* Black text for buttons */
            background-color: white !important;  /* White background for buttons */
            border: 1px solid #ddd !important;
            border-radius: 5px;
        }}
        .stButton > button:hover {{
            background-color: #f0f0f0 !important;  /* Slightly darker hover effect */
        }}
        .stSuccess, .stAlert {{
            background-color: white !important;  /* White background for result boxes */
            color: black !important;  /* Black text for result boxes */
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
def load_model(modelfile):
    with open(modelfile, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model
def input_page():
    html_temp = """
    <div>
    <h1 style="text-align:center;">Crop Recommendation System 🌱</h1>
    <p style="text-align:center; font-size:22px;">Find out the most suitable crop to grow in your farm 👨‍🌾</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    set_background_image("bg.jpg")

    N = st.number_input("Nitrogen (N)", 1, 10000)
    P = st.number_input("Phosphorus (P)", 1, 10000)
    K = st.number_input("Potassium (K)", 1, 10000)
    temp = st.number_input("Temperature (°C)", 0.0, 100.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("ph", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0)

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    if st.button('Predict'):
        loaded_model = load_model('model.pkl')
        prediction = loaded_model.predict(single_pred)
        st.session_state.prediction = prediction.item().title()
        st.session_state.page = "Result Page"

def result_page():
    html_temp = """
    <div>
    <h1 style="text-align:center;">Crop Recommendation Result 🌱</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    set_background_image("bg.jpg")
    if 'prediction' in st.session_state:
        st.write("### Recommended Crop")
        st.success(f"**{st.session_state.prediction}** is recommended for your farm.")
    else:
        st.write("Please input the data to get the recommendation.")
    if st.button('Go Back'):
        st.session_state.page = "Input Page"

def main():

    st.set_page_config(page_title="Crop Recommendation System 🌱", layout='centered', initial_sidebar_state="collapsed")

    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'page' not in st.session_state:
        st.session_state.page = "Input Page"

    if st.session_state.page == "Input Page":
        input_page()
    elif st.session_state.page == "Result Page":
        result_page()


if __name__ == '__main__':
    main()
