import pickle
import pandas as pd
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ✅ Define the exact features used during training
FEATURE_NAMES = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Sex_male',
    'Embarked_Q',
    'Embarked_S'
]

def predict_survival(input_data: dict):
    input_df = pd.DataFrame([input_data])

    # Add any missing columns
    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep correct order
    input_df = input_df[FEATURE_NAMES]

    prediction = model.p