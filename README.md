# IRIS-ML-Pipeline

A complete End-to-End Machine Learning Operations (MLOps) project. This repository demonstrates how to train a model on the Iris dataset, containerize the application, and deploy it to a Kubernetes cluster using CI/CD workflows.

## 📂 Project Structure

```text
├── .github/workflows   # CI/CD pipelines for automated testing and deployment
├── app/                # Main application source code
│   ├── artifacts/      # Stores trained model files (e.g., .pkl)
│   ├── data/           # Raw dataset storage
│   └── main.py         # Inference API application entry point
├── k8s/                # Kubernetes deployment manifests
├── build_push.sh       # Shell script to build Docker image and push to registry
├── Dockerfile          # Instructions to containerize the application
├── post.lua            # Lua script (For load testing)
├── requirements.txt    # Python dependencies
├── train.py            # Script to train the ML model
└── README.md           # Project documentation

