import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
scaler = load_scaler()

# Streamlit UI Design
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.image("789.jpg", width=200)

st.title("💰 Loan Approval Prediction System")
st.write("Fill in the details below to check your loan eligibility.")

# User name input
user_name = st.text_input("Enter your Name")

# Sidebar for inputs
st.sidebar.header("User Input Features")
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0.0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0.0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0.0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term (in days)", min_value=0.0)
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
expected_features = scaler.n_features_in_
if input_data.shape[1] != expected_features:
    input_data = input_data[:, :expected_features]  # Adjust feature count if necessary

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button("Predict Loan Approval"):
    if user_name.strip() == "":
        st.warning("Please enter your name before proceeding.")
    else:
        prediction = model.predict(input_data_scaled)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        st.success(f"{user_name}, your loan application is **{result}**! 🎉")
