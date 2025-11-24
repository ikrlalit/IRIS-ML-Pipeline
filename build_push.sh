#!/bin/bash

# train and build
python train.py

# confirm model exists
if [ ! -f app/artifacts/model.joblib ]; then
  echo "Model not found! Training failed."
  exit 1
fi

sudo docker build -t gcr.io/zeta-yen-459809-k6/iris-api:latest .
sudo docker push gcr.io/zeta-yen-459809-k6/iris-api:latest
