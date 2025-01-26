import joblib
import streamlit as st
import pandas as pd


model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')


st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #1DB954;
        color: #000000;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ED760;
    }
    .stMarkdown h1 {
        color: #1DB954;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown h2 {
        color: #1DB954;
        font-family: 'Arial', sans-serif;
    }
    .stNumberInput>div>div>input {
        background-color: #333333;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #1DB954;
    }
    .stSelectbox>div>div>div {
        background-color: #333333;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #1DB954;
    }
    .stSidebar {
        background-color: #111111;
        color: #ffffff;
    }
    .stSuccess {
        color: #1DB954;
    }
    .stWarning {
        color: #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🚀 SVM Prediction App")
st.markdown("Welcome to the **SVM Prediction App**! Enter the input features below to predict whether a user will purchase a product or not.")


st.sidebar.title("About the App")
st.sidebar.markdown("""
This app uses a **Support Vector Machine (SVM)** model to predict whether a user will purchase a product based on their:
- **Gender**
- **Age**
- **Estimated Salary**

The model was trained by Ali Khalid F2021266006.
""")


st.header("📝 Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])  
with col2:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)  

with col3:
    salary = st.number_input("Estimated Salary", min_value=0, value=50000) 

gender_encoded = 1 if gender == "Male" else 0

input_data = pd.DataFrame({
    'Gender': [gender_encoded],
    'Age': [age],
    'EstimatedSalary': [salary]
})

# Scale the input data
input_data_scaled = scaler.transform(input_data)


if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("🎉 Prediction: **Purchased**")
    else:
        st.warning("🚫 Prediction: **Not Purchased**")


st.markdown("---")
st.markdown("© 2025 SVM Prediction App. All rights reserved.")
