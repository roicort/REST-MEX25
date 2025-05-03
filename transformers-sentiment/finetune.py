
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import sys
sys.path.append('../')

from utils.metrics import RestMexMetrics
from utils.config import setConfig

device = setConfig()
metrics = RestMexMetrics()

df = pd.read_csv(r'../data/train/train.csv')
audf = pd.read_csv(r'../data/augmented/train.csv')
data = pd.concat([df, audf], ignore_index=True)

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
    per_device_train_batch_size=256,
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
tokenizer.save_pretrained("./")
trainer.evaluate()
