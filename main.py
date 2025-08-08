# File: main.py
# This is the core of our project: the FastAPI application that serves the model.
#
# 1. It loads the pre-trained model from the .joblib file.
# 2. It defines a Pydantic model for the request body.
# 3. It creates a "/predict" endpoint that takes text input, uses the model
#    to predict the sentiment, and returns the result.

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Initialize the FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API to predict sentiment (positive/negative) from text.",
    version="1.0"
)

# --- Pydantic Models for Input and Output Data ---
# This defines the structure of the request body for the /predict endpoint.
# It expects a JSON object with a single key "text" of type string.
class TextInput(BaseModel):
    text: str

# This defines the structure of the JSON response.
# By defining a specific response model, the API's contract is clear
# to developers and tools like the interactive documentation.
class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float

# --- Loading the Model ---
# Load the trained model pipeline when the application starts.
# This is more efficient than loading it for every request.
model_path = Path("sentiment_model.joblib")

if not model_path.exists():
    # This is a critical check. The API cannot run without the model file.
    # The user must run model.py first to generate it.
    raise RuntimeError("Model file 'sentiment_model.joblib' not found. Please run model.py to train and save the model first.")

# Load the model from the file.
model = joblib.load(model_path)
print("Model loaded successfully.")


# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    """A root endpoint to confirm that the API is running."""
    return {"message": "Welcome to the Sentiment Analysis API. Use the /predict endpoint to make predictions."}

@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
def predict_sentiment(input_data: TextInput):
    """
    Predicts the sentiment of a given text.
    - **text**: The input text to analyze.
    - **returns**: A JSON object with the predicted label ('positive' or 'negative')
                 and the prediction confidence score.
    """
    try:
        # The input text is accessed from the Pydantic model.
        text_to_predict = input_data.text

        # The model's .predict() method expects a list or iterable of texts.
        prediction_array = model.predict([text_to_predict])
        # The result is an array, so we get the first element.
        prediction = prediction_array[0]

        # The model's .predict_proba() gives confidence scores for each class.
        # It returns a 2D array: [[prob_class_0, prob_class_1]]
        probabilities = model.predict_proba([text_to_predict])
        confidence = probabilities[0][prediction]

        # Convert the numerical prediction (0 or 1) to a human-readable label.
        label = "positive" if prediction == 1 else "negative"

        return {
            "text": text_to_predict,
            "prediction": label,
            "confidence": float(confidence)
        }
    except Exception as e:
        # Basic error handling for any unexpected issues during prediction.
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")