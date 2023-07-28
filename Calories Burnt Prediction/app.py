import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
# Load the CSS file for styling
css = """
h1 {
    color: #0066cc;
}

button {
    background-color: #0066cc;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background-color: #0052a3;
}
"""

st.set_page_config(
    page_title="Calories Burned Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to make the prediction
def make_prediction(model, input_data):
    return model.predict(input_data)

# Load the XGBoost model
@st.cache_data()
def load_model():
    with open('calmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Main function for the app
#@st.cache_data
def main():
    # Set CSS
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # Load the XGBoost model
    model = load_model()

    # Title
    st.title("Calories Burned Prediction App")

    # Input form
    st.header("Enter the details:")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0
    age = st.number_input("Age", min_value=1, max_value=150, step=1)
    height = st.number_input("Height (cm)", min_value=1, max_value=300, step=1)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, step=0.1)
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=1440, step=1)
    heart_rate = st.number_input("Heart Rate", min_value=1, max_value=250, step=1)
    body_temp = st.number_input("Body Temperature (Celsius)", min_value=30.0, max_value=50.0, step=0.1)

    # Make prediction
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })

    if st.button("Predict Calories Burned"):
        prediction = make_prediction(model, input_data)
        st.success(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")

if __name__ == "__main__":
    main()