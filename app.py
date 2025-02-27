import streamlit as st
import numpy as np
import pickle
import base64
from PIL import Image

# Function to load the trained model and scaler
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
scaler = load_scaler()

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        base64_str = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{base64_str}");
                background-size: cover;
            }}
            .stButton > button {{
                background-color: #4CAF50;
                color: white;
                font-size: 20px;
                border-radius: 10px;
            }}
            .stTextInput, .stSelectbox, .stNumberInput {{
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                padding: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set Background Image (Use a local or hosted image)
set_background("background.jpg")  # Ensure you have an image named 'background.jpg' in the project folder

# Streamlit Page Configuration
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.image("789.jpg", width=200)  # Add a logo or branding image
st.title("💰 Loan Approval Prediction System")
st.write("Fill in the details below to check your loan eligibility.")

# User name input
user_name = st.text_input("Enter your Name", max_chars=50)

# Sidebar for user inputs
st.sidebar.header("User Input Features")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0.0, value=5000.0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0.0, value=0.0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=150.0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0)
Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert categorical inputs into numerical format
Gender = 1 if Gender == "Male" else 0
Married = 1 if Married == "Yes" else 0
Education = 1 if Education == "Graduate" else 0
Self_Employed = 1 if Self_Employed == "Yes" else 0
Dependents = 3 if Dependents == "3+" else int(Dependents)
Property_Area = 2 if Property_Area == "Urban" else (1 if Property_Area == "Semiurban" else 0)

# Create input data array
input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, 
                         CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])

# Ensure input matches expected features by the scaler
try:
    input_data_scaled = scaler.transform(input_data)
    
    # Prediction button
    if st.button("Predict Loan Approval"):
        if user_name.strip() == "":
            st.warning("Please enter your name before proceeding.")
        else:
            prediction = model.predict(input_data_scaled)[0]
            result = "Approved" if prediction == 1 else "Rejected"
            st.success(f"{user_name}, your loan application is **{result}**! 🎉")
except Exception as e:
    st.error(f"An error occurred: {e}. Please check your inputs.")
