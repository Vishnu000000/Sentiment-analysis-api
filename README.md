# Sentiment Analysis API

This project is a production-ready, containerized API for sentiment analysis. It takes a piece of text and predicts whether the sentiment is positive or negative.

This project demonstrates the ability to take a machine learning model from a training script to a deployed, scalable service—a critical skill for any ML Engineer.

---

## Features

- **RESTful API:** A clean, predictable API built with FastAPI.
- **ML Model Serving:** Serves a pre-trained Scikit-learn model for real-time predictions.
- **Containerized:** Packaged with Docker for easy deployment and scalability.
- **Input Validation:** Uses Pydantic for robust, type-checked API inputs.

---

## Tech Stack

- **Backend:**
  - **Python 3.9**
  - **FastAPI:** For building the high-performance API.
  - **Uvicorn:** As the ASGI server.
- **Machine Learning:**
  - **Scikit-learn:** For building and training the sentiment analysis model pipeline.
  - **Joblib:** For serializing and saving the trained model.
- **DevOps:**
  - **Docker:** For containerizing the entire application.

---

## How to Run

There are two ways to run this project: locally for development or using Docker for a production-like environment.

### 1. Running Locally

You'll need Python 3.7+ installed.

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/sentiment-analysis-api.git](https://github.com/YOUR_USERNAME/sentiment-analysis-api.git)
cd sentiment-analysis-api

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# STEP 1: Train and save the model
# This creates the sentiment_model.joblib file.
python model.py

# STEP 2: Run the API server
uvicorn main:app --reload
```

The API will be running at `http://localhost:8000`. You can access the interactive documentation at `http://localhost:8000/docs` to test the `/predict` endpoint.

### 2. Running with Docker (Recommended)

With Docker installed, you can build and run the entire application with two commands.

```bash
# Make sure you are in the project's root directory

# Build the Docker image. This will also run model.py inside the container.
docker build -t sentiment-api .

# Run the container
docker run -d -p 8000:8000 --name sentiment-container sentiment-api
```

The API will be running in a container, accessible at `http://localhost:8000`.

---

## Project Purpose

This project is a key portfolio piece for ML Engineering roles, demonstrating:

- **ML Model Lifecycle:** Understanding the end-to-end process from training to deployment.
- **API Development:** The ability to build robust APIs for serving models.
- **MLOps Principles:** Using containerization (Docker) to create reproducible and scalable ML services.
