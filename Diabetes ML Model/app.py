import pandas as pd
import streamlit as st
import pickle

# Load the classifier and scaler
with open('model.pkl', 'rb') as pkl:
    classifier = pickle.load(pkl)

with open('scaler.pkl', 'rb') as pkl:
    scaler = pickle.load(pkl)


def main():
    st.header("Diabetes Prediction")
    left, right = st.columns((2, 2))
    Pregnancies = left.number_input("Enter Pregnancies as whole number", step=1, value=0)
    Glucose = right.number_input("Enter Glucose as whole number", step=1, value=0)
    BloodPressure = left.number_input("Enter Blood Pressure as whole number", step=1, value=0)
    SkinThickness = right.number_input("Enter Skin Thickness as whole number", step=1, value=0)
    Insulin = left.number_input("Enter Insulin as whole number", step=1, value=0)
    BMI = right.number_input("Enter BMI as decimal number", step=1, value=0)
    DiabetesPedigreeFunction = left.number_input("Enter Diabetes Pedigree Function as decimal number", step=0.001, value=0.00)
    Age = right.number_input("Enter Age as Whole number", step=1, value=0)

    predict_button = st.button("Am I Diabetic??")

    if predict_button:
        # Create a pandas DataFrame from the input data
        data = pd.DataFrame({
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'SkinThickness': [SkinThickness],
            'Insulin': [Insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age]
        })

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(data)

        # Make predictions using the classifier
        result = classifier.predict(scaled_data)

        if result[0] == 0:
            st.success("You are not Diabetic")
        else:
            st.success("You are Diabetic")


if __name__ == "__main__":
    main()