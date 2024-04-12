#!/bin/bash

# Configuration
REPO_URL="https://github.com/miltonian/feature-transcribe-ai.git"
PROJECT_DIR="feature-transcribe-ai"
PYTHON_ENV_NAME="feature-transcribe-ai-py-env"
PROJECT_DIR="feature-transcribe-ai"

echo "Downloading and installing the project..."
# Clone the repository
git clone $REPO_URL $PROJECT_DIR
cd $PROJECT_DIR

# Check if Python 3 and pip are installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3."
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip."
    exit 1
fi

# Create a Python virtual environment
python3 -m venv $PYTHON_ENV_NAME

# Activate the virtual environment
source $PYTHON_ENV_NAME/bin/activate

# Install Python dependencies
pip3 install -r requirements.txt

echo "Installation complete. You can now run the project."

# Optional: Provide instructions to run the project
echo "To run the project, first you have to sync it:"
echo "source $PYTHON_ENV_NAME/bin/activate"
echo "export OPENAI_API_KEY=your-key"
echo "sh sync.sh"

echo "Then you can start running it:"
echo "sh run.sh"
