#!/bin/bash

# Install the required Python packages
echo "Installing Python packages..."
pip install --upgrade pip  # Upgrade pip to the latest version
pip install -r requirements.txt  # Install packages from requirements.txt

# Install NLTK resources
echo "Downloading NLTK resources..."
python -m nltk.downloader punkt punkt_tab

echo "Installation complete!"
