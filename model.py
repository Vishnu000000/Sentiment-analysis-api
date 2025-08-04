# File: model.py
# This script is responsible for the "training" part of our project.
# It uses the scikit-learn library to create a very simple sentiment analysis
# model. The key steps are:
# 1. Define some sample data.
# 2. Use a TF-IDF Vectorizer to convert text into numerical features.
# 3. Train a simple Logistic Regression classifier.
# 4. Bundle the vectorizer and the classifier into a single pipeline.
# 5. Save the trained pipeline to a file using joblib.

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_and_save_model():
    """
    Trains a simple sentiment analysis model and saves it to a file.
    """
    print("Training a simple sentiment analysis model...")

    # Sample data: a few positive and negative phrases.
    # In a real-world scenario, this would be a large dataset.
    X_train = [
        "I love this product, it's amazing",
        "This is the best purchase I've ever made",
        "Absolutely fantastic, highly recommend",
        "I am so happy with the service",
        "What a wonderful experience",
        "I hate this, it's terrible",
        "Worst experience ever, do not buy",
        "A complete waste of money and time",
        "I am very disappointed with the quality",
        "This is awful, I regret buying it"
    ]
    # Corresponding labels: 1 for positive, 0 for negative.
    y_train = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # Create a scikit-learn pipeline. This is a best practice for ML models.
    # It ensures that the same steps are applied to both training and prediction data.
    # Step 1: TfidfVectorizer - Converts text into a matrix of TF-IDF features.
    # Step 2: LogisticRegression - A simple and effective classification algorithm.
    sentiment_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    # Train the model on our sample data.
    sentiment_model.fit(X_train, y_train)
    
    # Define the filename for the saved model.
    model_filename = 'sentiment_model.joblib'
    
    # Save the trained pipeline to the specified file.
    # joblib is efficient for saving scikit-learn models.
    joblib.dump(sentiment_model, model_filename)

    print(f"Model trained and saved to '{model_filename}'")
    print("You can now run the main.py file to start the API server.")

if __name__ == "__main__":
    # This allows the script to be run directly from the command line.
    train_and_save_model()
