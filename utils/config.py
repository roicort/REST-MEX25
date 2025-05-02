import torch
import spacy
from tqdm import tqdm
import os

def setConfig():
    """
    Configura el dispositivo para PyTorch y spaCy según la disponibilidad de hardware.
    Devuelve el dispositivo configurado.
    """
    if torch.backends.mps.is_available():
        # Usar MPS (Metal Performance Shaders) en macOS
        device = torch.device("mps")
        spacy.prefer_gpu()
        print("Usando MPS:", device)
    elif torch.cuda.is_available():
        # Usar CUDA (GPU NVIDIA) si está disponible
        device = torch.device("cuda")
        spacy.prefer_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("Usando CUDA:", device)
    else:
        # Usar CPU como fallback
        device = torch.device("cpu")
        spacy.prefer_cpu()
        print("Usando CPU:", device)
    
    # Configurar TQDM en pandas
    tqdm.pandas()

    try:
        torch.ones(1, device=device)
        print("Test correcto:", device)
    except Exception as e:
        print("Error al configurar el dispositivo:", e)
        raise e
    
    return device