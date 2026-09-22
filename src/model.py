from transformers import pipeline
import os
from pathlib import Path
from src import newdata_manager
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset # libreria HF per la creazione dei dataset direttamente nel formato corretto per il training




source_model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = None
loaded_model_name = None

BASEMODEL_NAME = "BaseModel"

BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
MODEL_PATH = Path(
    os.getenv(
        "HF_MODEL_PATH",
        str(PROJECTDIR/"model")
    )
)


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
    )

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
            model= MODEL_PATH,
            tokenizer=MODEL_PATH
        )
    else:
        model_pipeline = pipeline(
            "sentiment-analysis",
            model= MODEL_PATH / model_version,
            tokenizer= MODEL_PATH / model_version
        )

    return model_pipeline

# funzioner per l'addestramento del modello - DA COMPLETARE

def train_model(dataset_name, new_version_name,  model_trained = BASEMODEL_NAME):

    dataset = newdata_manager.Dataset(dataset_name)

    X_train, X_test, y_train, y_test = dataset.return_training_data()

    X_train_tokenized = texts_tokenizer(X_train, model_trained)
    X_test_tokenized = texts_tokenizer(X_test, model_trained)

    X_train_tokenized["labels"] = y_train
    train_dataset = HFDataset.from_dict(X_train_tokenized)

    X_test_tokenized["labels"] = y_test
    test_dataset = HFDataset.from_dict(X_test_tokenized)


    return train_dataset, test_dataset




# funzione per tokenizzare il testo di un dataset

def texts_tokenizer(texts, model_name = BASEMODEL_NAME):

    model_path = MODEL_PATH / model_name

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
        padding = False
    
    )
    return encoded
