import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def predict_sentiment(texts, model, tokenizer, device, batch_size=16):
    dataloader = DataLoader(texts, batch_size=batch_size)
    predictions = []
    for batch in tqdm(dataloader, desc="Predicting sentiment"):
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Mover tensores a MPS
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        #sentiment_map = {1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}
        predictions.extend([int(p)+1 for p in torch.argmax(probabilities, dim=-1).tolist()])
    return predictions

def embed_texts(texts, model, tokenizer, device, batch_size=16, dimension=768):
    dataloader = DataLoader(texts, batch_size=batch_size)
    embeddings = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        inputs = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Mover tensores a MPS
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0][:dimension].cpu().numpy()  # Usar el token [CLS]
        embeddings.extend(batch_embeddings)
        del inputs, outputs, batch_embeddings
        if device == 'mps':
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
    return embeddings

def TokenizeText(text, nlp):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop]
    return tokens

def IdentityTokenizer(tokens):
    return tokens