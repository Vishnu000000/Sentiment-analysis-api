# File: Dockerfile
# This file packages our ML API into a container.

# Start with a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
# This means Docker won't reinstall packages unless this file changes.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (main.py, model.py)
COPY . .

# IMPORTANT: Run the model training script inside the Docker image during the build process.
# This ensures that the sentiment_model.joblib file is created and included in the final image.
RUN python model.py

# Expose the port the app runs on
EXPOSE 8000

# The command to run the FastAPI application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
