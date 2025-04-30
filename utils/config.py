import torch
import spacy
from tqdm import tqdm

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
        print("Usando CUDA:", device)
    else:
        # Usar CPU como fallback
        device = torch.device("cpu")
        spacy.prefer_cpu()
        print("Usando CPU:", device)
    
    # Configurar TQDM en pandas
    tqdm.pandas()

    # Crear un tensor de prueba para verificar el dispositivo
    _ = torch.ones(1, device=device)
    print("Tensor de prueba creado en el dispositivo:", _, device)
    return device