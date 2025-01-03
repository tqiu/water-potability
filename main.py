from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water

app = FastAPI(
    title="Water Potability Prediction",
    description="Predict whether water is potable or not",
    version="0.1",
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to Water Potability Prediction API"

@app.post("/predict")
def model_prediction(water: Water):
    data = pd.DataFrame([water.model_dump()])
    prediction = model.predict(data)
    if prediction == 1:
        return "Water is potable"
    else:
        return "Water is not potable"
