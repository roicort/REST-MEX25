
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
from metrics import RestMexMetrics
metrics = RestMexMetrics()
tqdm.pandas()

# Set up the device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv(r'./dataset/train.csv')

data['Title'] = data['Title'].astype(str)
data['Review'] = data['Review'].astype(str)
data['Town'] = data['Town'].astype(str)
data['Region'] = data['Region'].astype(str)
data['Type'] = data['Type'].astype(str)
data['Polarity'] = data['Polarity'].astype(int)

# Comenzar polaridad en 0 (Es decir 0,1,2,3,4)

data['Polarity'] = data['Polarity'].astype(int)
data['Polarity'] = data['Polarity'] - 1

train, test = train_test_split(data, test_size=0.15, random_state=42)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

datasetTrain = pd.DataFrame()
datasetTest = pd.DataFrame()

datasetTrain['text'] = '<title>' + train['Title'] + '</title> <review>' + train['Review'] + '</review>'
datasetTest['text'] = '<title>' + test['Title'] + '</title> <review>' + test['Review'] + '</review>'

datasetTrain['labels'] = train['Polarity'].astype(int)
datasetTest['labels'] = test['Polarity'].astype(int)

print(datasetTrain['labels'].unique())
print(datasetTest['labels'].unique())

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    return tokenized

dataset = DatasetDict({
    "train": Dataset.from_pandas(datasetTrain),
    "test": Dataset.from_pandas(datasetTest)
})

dataset = dataset.map(tokenize_function, batched=True, batch_size=512, remove_columns=['text'])

training_args = TrainingArguments(
    output_dir="./models/",
    eval_strategy="epoch",
    save_strategy="no",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./")
trainer.evaluate()
