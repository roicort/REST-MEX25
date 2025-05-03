import sys
import ast

import spacy
import joblib
import pandas as pd

from wasabi import msg
from radicli import Radicli

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report #, f1_score, accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification #, AutoModel

sys.path.append('../')
from utils.config import setConfig
from utils.metrics import RestMexMetrics
from utils.run import predict_sentiment, TokenizeText #, IdentityTokenizer, embed_texts

#### Global Configs ######################################

device = setConfig()
metrics = RestMexMetrics()
cli = Radicli()

#### CLI Commands ##########################################

@cli.command("eval_baseline")
def eval_baseline():
    """
    Evaluate the baseline ensamble.
    """

    #### Load Data #######################################

    data = pd.read_csv(r'../data/train/train_augmented_tokenized.csv', encoding='utf-8')

    data['Title'] = data['Title'].astype(str)
    data['Review'] = data['Review'].astype(str)
    data['Town'] = data['Town'].astype(str)
    data['Region'] = data['Region'].astype(str)
    data['Type'] = data['Type'].astype(str)
    data['Polarity'] = data['Polarity'].astype(int)

    data['Title_tokens'] = data['Title_tokens'].apply(ast.literal_eval)
    data['Review_tokens'] = data['Review_tokens'].apply(ast.literal_eval)

    _, test = train_test_split(data, test_size=0.20, random_state=42)

    #### Load Models #######################################
    
    best_type_model = joblib.load('./models/baseline/type_gridmodel.pkl').best_estimator_
    best_town_model = joblib.load('./models/baseline/magictown_gridmodel.pkl').best_estimator_
    best_polarity_model = joblib.load('./models/baseline/polarity_gridmodel.pkl').best_estimator_

    #### Predict Type ######################################

    Type_X_test = test['Title_tokens'] + test['Review_tokens']
    Type_y_test = test['Type']
    Type_y_test_pred = best_type_model.predict(Type_X_test)

    #### Predict Town ######################################

    Town_X_test = test['Region'].apply(lambda x: [str(x)]) + test['Title_tokens'] + test['Review_tokens']
    Town_y_test = test['Town']
    Town_y_test_pred = best_town_model.predict(Town_X_test)

    #### Predict Polarity ################################

    Polarity_X_test = test['Title_tokens'] + test['Review_tokens']
    Polarity_y_test = test['Polarity']
    Polarity_y_test_pred = best_polarity_model.predict(Polarity_X_test)

    #### Eval #############################################

    # Type
    type_report_md = classification_report(Type_y_test, Type_y_test_pred, target_names=test['Type'].unique(), output_dict=True)
    type_report_md = pd.DataFrame(type_report_md)
    f1 = type_report_md[['Attractive', 'Hotel', 'Restaurant']].loc['f1-score'].to_dict()
    ResT_k = metrics.TypeScore(f1)

    # Town
    town_report_md = classification_report(Town_y_test, Town_y_test_pred, target_names=test['Town'].unique(), output_dict=True)
    town_report_md = pd.DataFrame(town_report_md)
    f1 = town_report_md[Town_y_test.unique()].loc['f1-score'].to_dict()
    ResMT_k = metrics.TypeScore(f1)

    # Polarity
    polarity_report_md = classification_report(Polarity_y_test, Polarity_y_test_pred, target_names=test['Polarity'].astype(int).unique(), output_dict=True)
    polarity_report_md = pd.DataFrame(polarity_report_md)
    f1 = polarity_report_md[Polarity_y_test.unique()].loc['f1-score'].to_dict()
    ResP_k = metrics.TypeScore(f1)

    print(f"ResP_k: {ResP_k:.4f}")
    print(f"ResMT_k: {ResMT_k:.4f}")
    print(f"ResT_k: {ResT_k:.4f}")

    Sentiment_k = RestMexMetrics.RestMexScore(ResP_k, ResT_k, ResMT_k)
    print(f"Sentiment(k): {Sentiment_k:.4f}")

    #### Save Results ###############################

    # Guardar el reporte en formato Markdown
    report_md = f"""
    # Reporte de Métricas

    ## Métricas Calculadas

    - **ResP_k (Polarity Score):** {ResP_k:.4f}
    - **ResMT_k (Town Score):** {ResMT_k:.4f}
    - **ResT_k (Type Score):** {ResT_k:.4f}
    - **Sentiment(k) (Overall Score):** {Sentiment_k:.4f}

    ## Classification Reports

    ### Type Classification Report
    {type_report_md}

    ### Town Classification Report
    {town_report_md}

    ### Polarity Classification Report
    {polarity_report_md}

    """
    # Guardar el reporte en un archivo Markdown
    with open("./results/eval_baseline.md", "w") as file:
        file.write(report_md)

    #### END ###############################

@cli.command("eval_ensamble")
def eval_ensamble():
    """
    Evaluate the final ensamble.
    """

    #### Load Data #######################################

    df = pd.read_csv(r'./data/train/train.csv')
    audf = pd.read_csv(r'./data/augmented/train.csv')
    data = pd.concat([df, audf], ignore_index=True)

    data['Title'] = data['Title'].astype(str)
    data['Review'] = data['Review'].astype(str)
    data['Town'] = data['Town'].astype(str)
    data['Region'] = data['Region'].astype(str)
    data['Type'] = data['Type'].astype(str)
    data['Polarity'] = data['Polarity'].astype(int)

    _, test = train_test_split(data, test_size=0.20, random_state=42)

    #### Preprocess ###############################

    nlp = spacy.load("es_dep_news_trf")
    test['Title_tokens'] = test['Title'].progress_apply(lambda x: TokenizeText(x, nlp))
    test['Review_tokens'] = test['Review'].progress_apply(lambda x: TokenizeText(x, nlp))

    #### Load Models ###############################
    
    best_type_model = joblib.load('./models/baseline/type_gridmodel.pkl').best_estimator_
    best_town_model = joblib.load('./models/baseline/magictown_gridmodel.pkl').best_estimator_

    path = './models/tabularisai-distilbert/'
    model = AutoModelForSequenceClassification.from_pretrained(path) 
    tokenizer = AutoTokenizer.from_pretrained(path)

    #### Predict Polarity ################################

    Polarity_X_test = test['Title'] + test['Review']
    Polarity_y_test = test['Polarity']

    msg.info(f"Loaded model from {path}")

    Polarity_y_test_pred = []

    model = model.to(device)
    Polarity_y_test_pred = predict_sentiment(Polarity_X_test.to_numpy(), model, tokenizer, batch_size=16)

    msg.good("Polarity predictions done")

    #### Predict Type ######################################

    Type_X_test = test['Title_tokens'] + test['Review_tokens']
    Type_y_test = test['Type']
    Type_y_test_pred = best_type_model.predict(Type_X_test)

    #### Predict Town ######################################

    Town_X_test = test['Region'].apply(lambda x: [str(x)]) + test['Title_tokens'] + test['Review_tokens']
    Town_y_test = test['Town']
    Town_y_test_pred = best_town_model.predict(Town_X_test)

    #### Eval #############################################

    # Type
    type_report_md = classification_report(Type_y_test, Type_y_test_pred, target_names=test['Type'].unique(), output_dict=True)
    type_report_md = pd.DataFrame(type_report_md)
    f1 = type_report_md[['Attractive', 'Hotel', 'Restaurant']].loc['f1-score'].to_dict()
    ResT_k = metrics.TypeScore(f1)

    # Town
    town_report_md = classification_report(Town_y_test, Town_y_test_pred, target_names=test['Town'].unique(), output_dict=True)
    town_report_md = pd.DataFrame(town_report_md)
    f1 = town_report_md[Town_y_test.unique()].loc['f1-score'].to_dict()
    ResMT_k = metrics.TypeScore(f1)

    # Polarity
    polarity_report_md = classification_report(Polarity_y_test, Polarity_y_test_pred, target_names=test['Polarity'].astype(int).unique(), output_dict=True)
    polarity_report_md = pd.DataFrame(polarity_report_md)
    f1 = polarity_report_md[Polarity_y_test.unique()].loc['f1-score'].to_dict()
    ResP_k = metrics.TypeScore(f1)

    msg.info(f"ResP_k: {ResP_k:.4f}")
    msg.info(f"ResMT_k: {ResMT_k:.4f}")
    msg.info(f"ResT_k: {ResT_k:.4f}")

    Sentiment_k = RestMexMetrics.RestMexScore(ResP_k, ResT_k, ResMT_k)
    msg.good(f"Sentiment(k): {Sentiment_k:.4f}")

    #### Save Results ###############################

    # Guardar el reporte en formato Markdown
    report_md = f"""
    # Reporte de Métricas

    ## Métricas Calculadas

    - **ResP_k (Polarity Score):** {ResP_k:.4f}
    - **ResMT_k (Town Score):** {ResMT_k:.4f}
    - **ResT_k (Type Score):** {ResT_k:.4f}
    - **Sentiment(k) (Overall Score):** {Sentiment_k:.4f}

    ## Classification Reports

    ### Type Classification Report
    {type_report_md}

    ### Town Classification Report
    {town_report_md}

    ### Polarity Classification Report
    {polarity_report_md}

    """
    # Guardar el reporte en un archivo Markdown
    with open("./results/eval_baseline.md", "w") as file:
        file.write(report_md)

    #### END ###############################

@cli.command("eval_embeddings")
def eval_embeddings():
    """
    Evaluate the final ensamble.
    """

    #### Load Embeddings ###############################

    data = joblib.load('../data/train/embeddings.pkl')
    _, test = train_test_split(data, test_size=0.20, random_state=42)

    #### Load Models ###############################

    best_type_model = joblib.load('./models/embeddings/type_gridmodel.pkl').best_estimator_
    best_town_model = joblib.load('./models/embeddings/magictown_gridmodel.pkl').best_estimator_
    best_polarity_model = joblib.load('./models/embeddings/polarity_gridmodel.pkl').best_estimator_

    #### Predict Polarity ################################

    Polarity_X_test = test['embedding'].tolist()
    Polarity_y_test = test['Polarity']
    Polarity_y_test_pred = best_polarity_model.predict(Polarity_X_test)
    
    #### Predict Type ######################################

    Type_X_test = test['embedding'].tolist()
    Type_y_test = test['Type']
    Type_y_test_pred = best_type_model.predict(Type_X_test)

    #### Predict Town ######################################

    Town_X_test = test['embedding'].tolist()
    Town_y_test = test['Town']
    Town_y_test_pred = best_town_model.predict(Town_X_test)

    #### Eval ##############################################

    # Type
    type_report_md = classification_report(Type_y_test, Type_y_test_pred, target_names=test['Type'].unique(), output_dict=True)
    type_report_md = pd.DataFrame(type_report_md)
    f1 = type_report_md[['Attractive', 'Hotel', 'Restaurant']].loc['f1-score'].to_dict()
    ResT_k = metrics.TypeScore(f1)

    # Town
    town_report_md = classification_report(Town_y_test, Town_y_test_pred, target_names=test['Town'].unique(), output_dict=True)
    town_report_md = pd.DataFrame(town_report_md)
    f1 = town_report_md[Town_y_test.unique()].loc['f1-score'].to_dict()
    ResMT_k = metrics.TypeScore(f1)

    # Polarity
    polarity_report_md = classification_report(Polarity_y_test, Polarity_y_test_pred, target_names=test['Polarity'].astype(int).unique(), output_dict=True)
    polarity_report_md = pd.DataFrame(polarity_report_md)
    f1 = polarity_report_md[Polarity_y_test.unique()].loc['f1-score'].to_dict()
    ResP_k = metrics.TypeScore(f1)

    msg.info(f"ResP_k: {ResP_k:.4f}")
    msg.info(f"ResMT_k: {ResMT_k:.4f}")
    msg.info(f"ResT_k: {ResT_k:.4f}")
    Sentiment_k = RestMexMetrics.RestMexScore(ResP_k, ResT_k, ResMT_k)
    msg.good(f"Sentiment(k): {Sentiment_k:.4f}")

    #### Save Results ###############################
    # Guardar el reporte en formato Markdown
    report_md = f"""
    # Reporte de Métricas
    ## Métricas Calculadas
    - **ResP_k (Polarity Score):** {ResP_k:.4f}
    - **ResMT_k (Town Score):** {ResMT_k:.4f}
    - **ResT_k (Type Score):** {ResT_k:.4f}
    - **Sentiment(k) (Overall Score):** {Sentiment_k:.4f}
    ## Classification Reports
    ### Type Classification Report
    {type_report_md}
    ### Town Classification Report
    {town_report_md}
    ### Polarity Classification Report
    {polarity_report_md}
    """
    # Guardar el reporte en un archivo Markdown
    with open("./results/eval_embeddings.md", "w") as file:
        file.write(report_md)
    #### END ###############################

@cli.command("inference")
def inference():
    """
    Inference the final ensamble.
    """

    #### Load Data #######################################

    df = pd.read_csv(r'./data/test/test.csv', index_col=0)
    df['Title'] = df['Title'].astype(str)
    df['Review'] = df['Review'].astype(str)

    ##### Predict Polarity ################################

    Polarity_X_test = df['Title'] + df['Review']
    
    path = './models/tabularisai-distilbert/'
    model = AutoModelForSequenceClassification.from_pretrained(path) 
    tokenizer = AutoTokenizer.from_pretrained(path)

    msg.info(f"Loaded model from {path}")

    Polarity_y_test_pred = []

    model = model.to(device)
    Polarity_y_test_pred = predict_sentiment(Polarity_X_test.to_numpy(), model, tokenizer, batch_size=16)

    msg.good("Polarity predictions done")

    #### Preprocess ###############################

    nlp = spacy.load("es_dep_news_trf")
    df['Title_tokens'] = df['Title'].progress_apply(lambda x: TokenizeText(x, nlp))
    df['Review_tokens'] = df['Review'].progress_apply(lambda x: TokenizeText(x, nlp))

    #### Type #######################################

    Type_X_test = df['Title_tokens'] + df['Review_tokens']
    
    best_type_model = joblib.load('./models/baseline/type_gridmodel.pkl').best_estimator_
    
    Type_y_test_pred = best_type_model.predict(Type_X_test)

    #### Town #######################################

    Town_X_test = df['Region'].apply(lambda x: [str(x)]) + df['Title_tokens'] + df['Review_tokens']
    
    best_town_model = joblib.load('./models/baseline/magictown_gridmodel.pkl').best_estimator_
    
    Town_y_test_pred = best_town_model.predict(Town_X_test)

    #### Save Predictions ###############################

    submission_df = pd.DataFrame({
        'ID': df.index,
        'Title': df['Title'],
        'Review': df['Review'],
        'Type': Type_y_test_pred,
        'Town': Town_y_test_pred,
        'Polarity': Polarity_y_test_pred
    })

    submission_df.to_csv('./results/submission.csv', index=False)

    #### END ###############################

if __name__ == "__main__":
    cli.run()