#!/bin/bash

# Setup script for the project environment

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

setup_python() {
    echo "Setting up Python environment..."
    
    if ! command_exists python3; then
        echo "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi

    # Create virtual environment with access to system-site-packages if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment with system-site-packages..."
        python3 -m venv venv --system-site-packages
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install requirements
    python setup.py install

    echo "Python environment setup complete."
}

# Function to setup additional components
setup_additional() {
    echo "Setting up additional components..."

    # Create necessary directories
    mkdir -p data/{raw,processed,results}
    mkdir -p models
    mkdir -p logs

    # Set up any environment variables
    PROJECT_ROOT=$(pwd)
    export PROJECT_ROOT

    echo "Additional setup complete."
}

# Main setup function
main() {
    echo "Starting environment setup..."

    # Run setup functions
    setup_python
    setup_additional

    echo "Environment setup complete!"
    echo "To activate the Python environment, run: source venv/bin/activate"
}

# Run the main function
main
