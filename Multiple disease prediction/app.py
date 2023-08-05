import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open('model.pkl', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_disease_model.pkl', 'rb'))

# loading the saved StandardScaler objects
diabetes_scaler = pickle.load(open('scaler.pkl', 'rb'))
parkinsons_scaler = pickle.load(open('scaler_parkinsons.pkl', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

## Function to handle Diabetes Prediction
def diabetes_prediction():
    st.title('Diabetes Prediction using ML')
    
    st.subheader('Enter the following details:')
    pregnancies = st.text_input('Number of Pregnancies')
    glucose = st.text_input('Glucose Level')
    blood_pressure = st.text_input('Blood Pressure value')
    skin_thickness = st.text_input('Skin Thickness value')
    insulin = st.text_input('Insulin Level')
    bmi = st.text_input('BMI value')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value')
    age = st.text_input('Age of the Person')
    
    # Check if all input fields are not empty
    if pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and diabetes_pedigree_function and age:
        # preprocess input using StandardScaler
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
        input_data_scaled = diabetes_scaler.transform(input_data)

        # code for Prediction
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict(input_data_scaled)
            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
            else:
                st.success('The person is not diabetic')
    else:
        st.warning('Please fill in all the input fields.')


# Function to handle Heart Disease Prediction
def heart_disease_prediction():
    st.title('Heart Disease Prediction using ML')
    
    st.subheader('Enter the following details:')
    age = st.text_input('Age')
    sex = st.text_input('Sex (male=1, female=0)')
    cp = st.text_input('Chest Pain types (0: typical angina, 1: atypical angina; 2: non-anginal pain; 3: asymptomatic)')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = Yes; 0 = No)')
    restecg = st.text_input('Resting Electrocardiographic results (0: Normal; 1: Having ST-T wave abnormality; 2: Showing probable or definite left ventricular hypertrophy)')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment (values:0;1;2)')
    ca = st.text_input('Number of major vessels colored by flourosopy (0-3)')
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')
    
    # Check if all input fields are not empty
    if age and sex and cp and trestbps and chol and fbs and restecg and thalach and exang and oldpeak and slope and ca and thal:
        # Convert numeric input fields to float and categorical input fields to int
        age = float(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = float(trestbps)
        chol = float(chol)
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = float(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)
        
        # code for Prediction
        if st.button('Heart Disease Test Result'):
            heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            if heart_prediction[0] == 0:
                st.success('The Person does not have a Heart Disease')
            else:
                st.success('The Person has Heart Disease')
    else:
        st.warning('Please fill in all the input fields.')

# Function to handle Parkinson's Prediction
def parkinsons_prediction():
    st.title("Parkinson's Disease Prediction using ML")
    
    st.subheader('Enter the following details:')
    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    jitter_percent = st.text_input('MDVP:Jitter(%)')
    jitter_abs = st.text_input('MDVP:Jitter(Abs)')
    rap = st.text_input('MDVP:RAP')
    ppq = st.text_input('MDVP:PPQ')
    ddp = st.text_input('Jitter:DDP')
    shimmer = st.text_input('MDVP:Shimmer')
    shimmer_db = st.text_input('MDVP:Shimmer(dB)')
    apq3 = st.text_input('Shimmer:APQ3')
    apq5 = st.text_input('Shimmer:APQ5')
    apq = st.text_input('MDVP:APQ')
    dda = st.text_input('Shimmer:DDA')
    nhr = st.text_input('NHR')
    hnr = st.text_input('HNR')
    rpde = st.text_input('RPDE')
    dfa = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    d2 = st.text_input('D2')
    ppe = st.text_input('PPE')
    
    # Check if all input fields are not empty
    if fo and fhi and flo and jitter_percent and jitter_abs and rap and ppq and ddp and shimmer and shimmer_db and apq3 and apq5 and apq and dda and nhr and hnr and rpde and dfa and spread1 and spread2 and d2 and ppe:
        # preprocess input using StandardScaler
        input_data = [[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]]
        input_data_scaled = parkinsons_scaler.transform(input_data)

        # code for Prediction
        if st.button("Parkinson's Test Result"):
            parkinsons_prediction = parkinsons_model.predict(input_data_scaled)
            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")
    else:
        st.warning('Please fill in all the input fields.')


# Main section for page navigation
if selected == 'Diabetes Prediction':
    diabetes_prediction()
elif selected == 'Heart Disease Prediction':
    heart_disease_prediction()
elif selected == 'Parkinsons Prediction':
    parkinsons_prediction()