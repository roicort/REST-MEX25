# Tokenize

import sys
import pandas as pd
import spacy
from wasabi import msg
sys.path.append('../')
from utils.config import setConfig

device = setConfig()

df = pd.read_csv(r'../data/train/train.csv')
audf = pd.read_csv(r'../data/augmented/train.csv')
df = pd.concat([df, audf], ignore_index=True)

df['Title'] = df['Title'].astype(str)
df['Review'] = df['Review'].astype(str)
df['Town'] = df['Town'].astype(str)
df['Region'] = df['Region'].astype(str)
df['Type'] = df['Type'].astype(str)
df['Polarity'] = df['Polarity'].astype(int)

nlp = spacy.load("es_dep_news_trf")

def tokenize_text(text, nlp):
    """
    Tokenize the input text, removing punctuation, stopwords, and converting to lowercase.
    
    Args:
        text (str): The input text to tokenize.
        nlp (spacy.Language): The spaCy language model.
    
    Returns:
        list: A list of processed tokens.
    """
    # Procesar el texto con el modelo de spaCy
    doc = nlp(text)
    
    # Extraer tokens, eliminar puntuación, stopwords y convertir a minúsculas
    tokens = [token.text.lower() for token in doc if not token.is_stop]
    
    return tokens


# Tokenize data with progress bar
df['Title_tokens'] = df['Title'].progress_apply(lambda x: tokenize_text(x, nlp))
df['Review_tokens'] = df['Review'].progress_apply(lambda x: tokenize_text(x, nlp))

# Save the tokenized data to a CSV file

df.to_csv(r'../data/train/train_augmented_tokenized.csv', index=False)
msg.good("Tokenization completed")