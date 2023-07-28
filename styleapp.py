import pandas as pd
import streamlit as st
import pickle

# Load the classifier and scaler
with open('model.pkl', 'rb') as pkl:
    classifier = pickle.load(pkl)

with open('scaler.pkl', 'rb') as pkl:
    scaler = pickle.load(pkl)



def main():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            color: #333333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #0066cc;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 2.2rem;
            font-weight: bold;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .feature-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .custom-button {
            background-color: #0066cc;
            color: white;
            font-size: 1.2rem;
            padding: 10px 25px;
            border-radius: 5px;
            margin-top: 30px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .custom-button:hover {
            background-color: #0059b3;
        }
        .footer {
            color: #666666;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header">Vaccine Usage Prediction App</div>', unsafe_allow_html=True)
    st.subheader('Predict whether someone received the H1N1 vaccine')

    st.markdown('<div class="container">', unsafe_allow_html=True)

    st.write('Enter the values for the features below:')

    
    h1n1_worry = st.slider('H1N1 Worry', 0, 3, 1, key='h1n1_worry')
    h1n1_awareness = st.slider('H1N1 Awareness', 0, 2, 1, key='h1n1_awareness')
    dr_recc_h1n1_vacc = st.checkbox('Doctor Recommended H1N1 Vaccine', key='dr_recc_h1n1_vacc')
    dr_recc_seasonal_vacc = st.checkbox('Doctor Recommended Seasonal Flu Vaccine', key='dr_recc_seasonal_vacc')
    chronic_medic_condition = st.checkbox('Has Chronic Medical Condition', key='chronic_medic_condition')
    is_health_worker = st.checkbox('Is Health Worker', key='is_health_worker')
    is_h1n1_vacc_effective = st.select_slider('Perceived H1N1 Vaccine Effectiveness', options=[1, 2, 3, 4, 5], value=3, key='is_h1n1_vacc_effective')
    is_h1n1_risky= st.select_slider('Perceived H1N1 Risk', options=[1, 2, 3, 4, 5], value=3,key='is_h1n1_risky')
    is_seas_vacc_effective=st.select_slider('Perceived Seasonal Vaccine Effectiveness', options=[1, 2, 3, 4, 5], value=3,key='is_seas_vacc_effective')
    is_seas_risky=st.select_slider('Perceived Seasonal Flu Risk', options=[1, 2, 3, 4, 5], value=3,key='is_seas_risky')



    if st.button('Predict', key='predict_btn'):
        # Create a pandas DataFrame from the input data
        data = pd.DataFrame({
    'h1n1_worry': [h1n1_worry],
    'h1n1_awareness': [h1n1_awareness],
    'dr_recc_h1n1_vacc': [dr_recc_h1n1_vacc],
    'dr_recc_seasonal_vacc': [dr_recc_seasonal_vacc],
    'chronic_medic_condition': [chronic_medic_condition],
    'is_health_worker': [is_health_worker],
    'is_h1n1_vacc_effective': [is_h1n1_vacc_effective],
    'is_h1n1_risky': [is_h1n1_risky],
    'is_seas_vacc_effective': [is_seas_vacc_effective],
    'is_seas_risky': [is_seas_risky]
})

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(data)

        # Make predictions using the classifier
        result = classifier.predict(scaled_data)

        if result[0] == 1:
            st.write("The respondent received the H1N1 vaccine.")
            st.image('H1N1vaccine.jpg', use_column_width=True, caption='Vaccinated!')
        else:
            st.write("The respondent did not receive the H1N1 vaccine.")
            st.image('noH1N1vaccine.jpg', use_column_width=True, caption='Not Vaccinated!')

    st.markdown('</div>', unsafe_allow_html=True)

    # Add a footer
    st.markdown(
        """
        <hr>
        <p class="footer">Built by Harikrishnan Nair. | Powered by Streamlit</p>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()