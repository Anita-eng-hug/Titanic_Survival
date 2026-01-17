import pickle
import pandas as pd
import os

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model, feature_names = pickle.load(f)

def predict_survival(input_data: dict):
    """
    input_data: dictionary with passenger features
    Example:
    {
        'Pclass': 3,
        'Age': 29,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 7.25,
        'Sex_male': 1,
        'Embarked_Q': 0,
        'Embarked_S': 1
    }
    """
    input_df = pd.DataFrame([input_data])

    # Ensure all columns exist
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, round(probability, 2)