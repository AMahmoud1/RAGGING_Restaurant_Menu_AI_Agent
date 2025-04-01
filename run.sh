#!/bin/bash

# Conda venv name
VENV_NAME="restaurant_ai_agent_venv"

# Handbook PDF Path
PDF_PATH="Ujamaa-restaurant-menu.pdf"

# Define model variables
LLM_MODEL="llama3.1:8b"

# FAISS Settings
FAISS_INDEX_PATH="faiss_index.index"
FIASS_k=3  # Number of similar returned chunks

# RAGGING Settings
NUM_QUERIES=4
CHUNKS_FILES_PATH="chunks"
MQR_TEMP=0.3
FINAL_RESPONSE_TMP=0.7

# Function to check if Conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
    else
        echo "Conda is not installed. Please install Conda before running this script."
        exit 1
    fi
}

# Function to check if Ollama is installed
check_ollama() {
    if command -v ollama &> /dev/null; then
        echo "Ollama is already installed."
    else
        echo "Ollama is not installed. Installing now..."
        curl -fsSL https://ollama.com/install.sh | sh
        if command -v ollama &> /dev/null; then
            echo "Ollama installed successfully."
        else
            echo "Failed to install Ollama. Please check the installation script or your network connection."
            exit 1
        fi
    fi
}

# Function to set up Conda virtual environment
setup_conda_env() {
    if conda env list | grep -q "^$CONDA_ENV_NAME\s"; then
        echo "Conda environment '$CONDA_ENV_NAME' already exists."
    else
        echo "Creating Conda environment: $CONDA_ENV_NAME..."
        conda create -n "$CONDA_ENV_NAME" -y
        echo "Conda environment '$CONDA_ENV_NAME' created successfully."
    fi

    # Activate the Conda environment
    echo "Activating Conda environment: $CONDA_ENV_NAME..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"

    # Install required packages
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install -Uqqq pip --progress-bar off
    pip install -qqq ollama==0.4.6 --progress-bar off
    pip install -qqq textsplitter --progress-bar off
    pip install -qqq PyPDF2 sentence faiss-cpu --progress-bar off
    pip install -qqq transformers --progress-bar off
    pip install -qqq faiss-cpu --progress-bar off
    pip install -qqq langchain==0.1.14 langchain-experimental==0.0.56
    pip install -Uqqq langchain-community==0.0.31
    pip install -qqq pdfplumber sentence-transformers streamlit torch openpyxl
    pip install -qqq bert-score
    echo "All required packages installed successfully."
}

# Run the function
check_conda
check_ollama
setup_conda_env

# Run Ollama serve in the background
echo "Starting Ollama serve in the background..."
ollama serve &

# Get the process ID and confirm it's running
OLLAMA_PID=$!
echo "Ollama serve is running in the background with PID: $OLLAMA_PID"

# Pull deepseek-r1 model
echo "Pulling LLM Models."
ollama pull "$LLM_MODEL"
echo "Done pulling LLM Models"

# Determine which Python script to run (run_chatbot.sh -t)
echo "Running main script..."
streamlit run main.py "$LLM_MODEL" "$PDF_PATH" "$FAISS_INDEX_PATH" "$FIASS_k" "$NUM_QUERIES" "$CHUNKS_FILES_PATH" "$MQR_TEMP" "$FINAL_RESPONSE_TMP"
echo "All tasks completed successfully."
conda deactivate