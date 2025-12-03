#!/bin/bash
# setup.sh

# Set Python version
export PYTHON_VERSION=3.12

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt