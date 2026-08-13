from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI application
app = FastAPI()

# Load the trained Machine Learning model
model = joblib.load("model.pkl")


# Define the input data structure
class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "ML API Running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(flower: Flower):

    # Convert input data into the format expected by the model
    sample = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    # Run inference
    prediction = model.predict(sample)

    # Return the prediction as JSON
    return {
        "prediction": int(prediction[0])
