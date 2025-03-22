#!/bin/bash
# CPU-optimized Neural Architecture Search for CASPAR model

# Use TensorFlow CPU optimization (optional, if installed with CPU optimizations)
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=1
export AWS_ACCESS_KEY_ID=mlops-demo
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_SECRET_ACCESS_KEY=mlops-demo
export MLFLOW_TRACKING_URI=http://localhost:9002

# Set the number of intra-op and inter-op threads for TensorFlow
# This helps with CPU parallelization
# Recommended to set these to the number of physical cores
export TF_NUM_INTRAOP_THREADS=12
export TF_NUM_INTEROP_THREADS=12

# Run a small-scale NAS optimized for CPU
echo "Running CPU-optimized Neural Architecture Search..."

python -m caspar.__main__ --hidden-layers 354 354 --dropout-rate 0.06 --batch 34 --epochs 60 --learning-rate 0.01
