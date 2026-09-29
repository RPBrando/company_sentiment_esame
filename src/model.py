from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
from pathlib import Path
from src import newdata_manager
from datasets import Dataset as HFDataset # libreria HF per la creazione dei dataset direttamente nel formato corretto per il training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



# origine del modello
source_model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

# variabili globali
sentiment_pipeline = None
loaded_model_name = None
tokenizer = None

# nome modello base

BASEMODEL_NAME = "BaseModel"

# percorsi utili

BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
MODEL_PATH = Path(
    os.getenv(
        "HF_MODEL_PATH",
        str(PROJECTDIR/"model")
    )
)

# recupero della pipeline per il modello scelto

def get_sentiment_pipeline(model_name = BASEMODEL_NAME):

    new_model_path = MODEL_PATH / model_name

# controllo che sia già stato importato il modello

    global sentiment_pipeline, loaded_model_name

    if sentiment_pipeline is None:
        if model_name == BASEMODEL_NAME:

            if not new_model_path.exists():
                sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=source_model_path)
                
                sentiment_pipeline.save_pretrained(
                    new_model_path
                )
                loaded_model_name = BASEMODEL_NAME
            else:
                if not (new_model_path / "config.json").exists():
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis", 
                        model=source_model_path)
                    sentiment_pipeline.save_pretrained(
                        new_model_path
                    )
                    loaded_model_name = BASEMODEL_NAME
                else:
                    sentiment_pipeline = load_saved_model(model_name)
                    loaded_model_name = BASEMODEL_NAME
        else:
            if not new_model_path.exists():
                sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=source_model_path)
                print("MODELLO INESISTENTE. Caricato modello base")
                loaded_model_name = BASEMODEL_NAME
            else:
                if not (new_model_path / "config.json").exists():
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis", 
                        model=source_model_path)
                    print("CARTELLA MODELLO VUOTA O CORROTTA. Caricato modello base")
                    loaded_model_name = BASEMODEL_NAME
                else:
                    sentiment_pipeline = load_saved_model(model_name)
                    loaded_model_name = model_name
                    
    elif loaded_model_name != model_name:
        sentiment_pipeline = change_used_model(model_name)    


    return sentiment_pipeline

# cambio del modello utilizzato

def change_used_model(model_path):
    global sentiment_pipeline, loaded_model_name
    
    if sentiment_pipeline is not None:
        if loaded_model_name != model_path:
            sentiment_pipeline = None
            sentiment_pipeline = get_sentiment_pipeline(model_path)

    return sentiment_pipeline

def make_prediction(text, batch_size=16, model_path = BASEMODEL_NAME):
    sentiment_pipeline = get_sentiment_pipeline(model_path)

    # text puo' essere una singola stringa oppure una lista di stringhe.
    # batch_size divide le liste grandi in blocchi piu' piccoli, cosi'
    # il modello non prova a processare troppi testi tutti insieme.
    return sentiment_pipeline(
        text,
        batch_size=batch_size,
        # Alcuni testi del dataset possono essere lunghi.
        # truncation=True li taglia alla lunghezza massima supportata dal modello.
        truncation=True,
        max_length=512
    ), loaded_model_name

def save_model(model_version = BASEMODEL_NAME):
    """
    model_version = BASEMODEL_PATH -> standard
    Se model_version != da None il modello viene salvato nella
    cartella della versione iniziale
    """
    global loaded_model_name, sentiment_pipeline
    
    if model_version == BASEMODEL_NAME:
        sentiment_pipeline = get_sentiment_pipeline(model_version)
    else:
        if (loaded_model_name != BASEMODEL_NAME) and (loaded_model_name == model_version):
            sentiment_pipeline.save_pretrained(MODEL_PATH / model_version)



def load_saved_model(model_version = None):
    """
    model_version = BASEMODEL_PATH -> standard
    Se model_version != da None il modello viene caricato dalla
    cartella della versione iniziale
    """


    if model_version is None:
        model_pipeline = pipeline(
            "sentiment-analysis",
            model= source_model_path,
        )
    else:
        model_pipeline = pipeline(
            "sentiment-analysis",
            model= MODEL_PATH / model_version,
            tokenizer= MODEL_PATH / model_version
        )

    return model_pipeline

# funzione per la tokenizzazione dei testi

def texts_tokenizer(texts, model_name = BASEMODEL_NAME):

    model_path = MODEL_PATH / model_name
    global tokenizer

    # controllo se il modello scelto è funzionante

    if model_name == BASEMODEL_NAME:
        tokenizer_source = source_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    else:
        if model_path.exists():
            if (model_path / "config.json").exists():
                tokenizer_source = MODEL_PATH / model_name
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            else:
                print("Il modello inserito nonn è esistente")
                return None
        else:
            print("Il modello inserito non è esistente")
            return None

    encoded = tokenizer(
        texts,
        truncation = True,
        max_length = 512,
        padding = True
    
    )
    return encoded


# definizione delle metriche di valutazione

def compute_metrics(eval_prediction):

# eval_prediction è una tupla che contiene due elementi:
# i valori predetti dal modello (logits)
# e i valori reali del dataset di test (labels)
# questi valori vengono passati dal trainer di HF per calcolare le metriche 
# di valutazione direttamente durante il training e la valutazione del modello

    logits, labels = eval_prediction 

# prende l'indice del valore massimo per ogni riga, 
# che corrisponde alla classe predetta

    predictions = logits.argmax(axis=-1) 

# calcolo delle metriche di valutazione

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )
# calcolo dell'accuracy

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# funzioner per l'addestramento del modello

def train_model(dataset_name, new_version_name, model_trained = BASEMODEL_NAME, sample_size = None):

    metrics = {}
    new_model_path = MODEL_PATH / new_version_name

    dataset = newdata_manager.Dataset(dataset_name)

    X_train, X_test, y_train, y_test = dataset.return_training_data(sample_size)

# creazione dei testi tokenizzati già separati per train e test

    X_train_tokenized = texts_tokenizer(X_train, model_trained)
    X_test_tokenized = texts_tokenizer(X_test, model_trained)

    global tokenizer

# creazione dataset train e test con libreria datasets
# rendiamo i dati pronti per il trainer

    X_train_tokenized["labels"] = y_train
    train_dataset = HFDataset.from_dict(X_train_tokenized)

    X_test_tokenized["labels"] = y_test
    test_dataset = HFDataset.from_dict(X_test_tokenized)

# rendiamo il modello caricabile a Trainer

    if model_trained == BASEMODEL_NAME:
        traning_model = AutoModelForSequenceClassification.from_pretrained(
            source_model_path
        )
    else:
        traning_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH / model_trained
        )

# definizione dei parametri di training

    arguments = TrainingArguments(
        output_dir= str(MODEL_PATH / "training_arguments" / new_version_name),
        num_train_epochs= 1,
        per_device_train_batch_size= 8,
        per_device_eval_batch_size= 8,
        learning_rate= 2e-5,
        save_strategy= "no",
        report_to= "none"
    )


# creazione del trainer

    trainer = Trainer(
        model =traning_model,
        train_dataset= train_dataset,
        eval_dataset= test_dataset,
        args= arguments,
        compute_metrics= compute_metrics,
    )

# valutazione del modello prima e dopo l'addestramento

    metrics["before training"] = trainer.evaluate(
        metric_key_prefix= "before_training"
    )
# addestramento del modello

    trainer.train()

# valutazione del modello dopo l'addestramento

    metrics["after training"] = trainer.evaluate(
        metric_key_prefix= "after_training"
    )

    trained_model = trainer.model

# salvataggio del nuovo modello

    trainer.save_model(new_model_path)
    tokenizer.save_pretrained(new_model_path)

# controllo che il modello sia stato salvato correttamente

    if new_model_path.exists():
        if (new_model_path/"config.json").exists():
            print("modello salvato correttamente")
        else:
            print("errore nel salvataggio del modello")
    else:
        print("errore nel salvataggio del modello")

    return trained_model, metrics