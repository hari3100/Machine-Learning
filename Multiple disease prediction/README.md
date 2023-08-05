Multiple Disease Prediction App

This project is a web-based application built using Streamlit that predicts three different diseases: diabetes, heart disease, and Parkinson's disease. The app allows users to input relevant medical information, and based on the trained machine learning models, it provides predictions for each disease.

Features

Predicts diabetes, heart disease, and Parkinson's disease separately using specific machine learning models for each condition.
User-friendly interface with a sidebar to select the disease of interest.
Displays dataset information for each disease for transparency and data understanding.
Utilizes standard scaling for input data when required (used for diabetes and Parkinson's disease prediction).
Trained models are stored in pickle files and are loaded when making predictions.

Technologies Used

Python: Programming language used for model training and app development.
Streamlit: Web application framework used for building the user interface.
Scikit-learn: Library used for machine learning model training.
Pandas: Library used for data handling and manipulation.

Models Used

Diabetes Prediction: Logistic Regression (trained using the StandardScaler)
Heart Disease Prediction: Logistic Regression
Parkinson's Disease Prediction: Support Vector Machine (SVM) (trained using the StandardScaler)

How to Run the App

Clone the repository to your local machine.
Install the required dependencies using pip install -r requirements.txt.
Ensure that the pickle files for trained models (standardScalar_diabetes.pkl, diabetes_model.pkl, heart_disease_model.pkl, scaler_parkinsons.pkl, parkinsons_disease_model.pkl) are present in the same directory as the app.py file.
Run the app using streamlit run app.py.
The app will open in your web browser, and you can start making predictions for different diseases.

Future Improvements

Include more disease prediction models to expand the application's scope.
Enhance the UI with additional visualizations and user interaction.
Deploy the app on a cloud platform for easy access.
Feel free to customize the app, add more features, and contribute to this open-source project. Your feedback and suggestions are welcome!

For any questions or assistance, please feel free to contact Harikrishnan Nair, harrinair2000@gmail.com
