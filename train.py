import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
from google.cloud import aiplatform

#First set Google Cloud project information
PROJECT_ID = "zeta-yen-459809-k6"
LOCATION = "us-central1"
BUCKET_URI = f"gs://mlops-course-zeta-yen-459809-k6-unique"
MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-1"
REPOSITORY = "iris-classifier-repo"
IMAGE = "iris-classifier-img"
MODEL_DISPLAY_NAME = "iris-classifier"

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

# Load and prepare the Iris dataset
def load_and_prepare_data():
    data = pd.read_csv('app/data/iris.csv')
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species
    return X_train, y_train, X_test, y_test

# Train the Decision Tree model
def train_model(X_train, y_train):
    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    return mod_dt

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy))
    return accuracy

# Create bucket if it doesn't exist and upload model artifacts to Cloud Storage
def save_and_upload_model(model, bucket_uri, model_artifact_dir):
    # Save the model
    joblib.dump(model, "app/artifacts/model.joblib")
    
    # Create bucket if it doesn't exist
    create_bucket_cmd = f"gsutil mb -l {LOCATION} -p {PROJECT_ID} {bucket_uri}"
    exit_code = os.system(create_bucket_cmd)
    if exit_code == 0:
        print(f"Bucket {bucket_uri} created or already exists")
    else:
        print(f"Failed to create bucket {bucket_uri}, it may already exist or there was an error")
    
    # Upload model artifacts to Cloud Storage
    upload_cmd = f"gsutil cp app/artifacts/model.joblib {bucket_uri}/{model_artifact_dir}/"
    upload_exit_code = os.system(upload_cmd)
    if upload_exit_code == 0:
        print(f"Model artifacts uploaded to {bucket_uri}/{model_artifact_dir}/")
    else:
        print(f"Failed to upload model artifacts to {bucket_uri}/{model_artifact_dir}/")

def main():
    # Load and prepare data
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save and upload model artifacts
    save_and_upload_model(model, BUCKET_URI, MODEL_ARTIFACT_DIR)

if __name__ == "__main__":
    main()
