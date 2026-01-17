import streamlit as st
from pp import predict_survival

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability.")

# Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
age = st.number_input("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)

sex = st.radio("Sex", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs for model
sex_male = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = {
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}

if st.button("Predict Survival"):
    prediction, probability = predict_survival(input_data)
    if prediction == 1:
        st.success(f"The passenger is predicted to survive 🚢 (Probability: {probability*100}%)")
    else:
        st.error(f"The passenger is predicted NOT to survive ☠️ (Probability: {probability*100}%)")
