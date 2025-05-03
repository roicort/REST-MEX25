#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip for Python 3."
    exit 1
fi

# Install requirements if requirements.txt exists
if [ -f "environment.yml" ]; then
    echo "Installing dependencies in the base environment..."
    conda env update -n base -f environment.yml || { echo "Failed to install requirements in the base environment."; exit 1; }
else
    echo "environment.yml not found. Installing default dependencies in the base environment..."
fi

# Check the operating system
echo "Installing Pip dependencies..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS."
    echo "Installing PyTorch for macOS..."
    pip install torch torchvision torchaudio || { echo "Failed to install PyTorch."; exit 1; }
    pip install -U pip setuptools wheel || { echo "Failed to update pip, setuptools, or wheel."; exit 1; }
    pip install -U 'spacy[apple]' || { echo "Failed to install spaCy for macOS."; exit 1; }
    pip install transformers[torch] || { echo "Failed to install transformers."; exit 1; }

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Ubuntu/Linux."
    # Check for CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -n 1)
        echo "Detected CUDA version: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == "12.8" ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || { echo "Failed to install PyTorch for CUDA 12.8."; exit 1; }
        elif [[ "$CUDA_VERSION" == "12.6" ]]; then
            pip install torch torchvision torchaudio || { echo "Failed to install PyTorch for CUDA 12.6."; exit 1; }
        else
            echo "Unsupported CUDA version or no specific version found. Installing CPU version of PyTorch."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "Failed to install CPU version of PyTorch."; exit 1; }
        fi

        pip install -U pip setuptools wheel || { echo "Failed to update pip, setuptools, or wheel."; exit 1; }
        pip install -U 'spacy[cuda12x]' || { echo "Failed to install spaCy for CUDA."; exit 1; }
        pip install transformers[torch] || { echo "Failed to install transformers."; exit 1; }

    else
        echo "CUDA not detected. Installing CPU version of PyTorch."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "Failed to install CPU version of PyTorch."; exit 1; }
        pip install -U pip setuptools wheel || { echo "Failed to update pip, setuptools, or wheel."; exit 1; }
        pip install -U spacy || { echo "Failed to install spaCy."; exit 1; }
    fi
else
    echo "Unsupported operating system."
    exit 1
fi

pip install transformers datasets evaluate || { echo "Failed to install transformers dataset & evaluate"; exit 1; }
python -m spacy download en_core_web_trf || { echo "Failed to download spaCy model en_core_web_trf."; exit 1; }
python -m spacy download es_dep_news_trf || { echo "Failed to download spaCy model es_dep_news_trf."; exit 1; }
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))" || { echo "Transformers test failed."; exit 1; }
echo "Installation complete."