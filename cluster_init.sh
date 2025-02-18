#!/bin/bash

# Log output for debugging
exec > /databricks/init_scripts/install_log.txt 2>&1

echo "Updating package list..."
sudo apt-get update -y

echo "Installing required packages..."
sudo apt-get install -y poppler-utils

echo "Installation complete."
