from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import os
from pathlib import Path
from src import newdata_manager
from datasets import (
    Dataset as HFDataset,
)  # libreria HF per la creazione dei dataset direttamente nel formato corretto per il training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src import csvfiles_manager
import pandas as pd


# origine del modello
source_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# variabile globale per il file di trigger per il retraining automatico del modello
trigger_file_path = csvfiles_manager.FILEPATH / "retrain_triggered" / "trigger.txt"
trigger_file_path.parent.mkdir(parents=True, exist_ok=True)

# variabili globali
sentiment_pipeline = None
loaded_model_name = None
tokenizer = None
best_model_name = None
retrain_model = False
stop_bad_retraining = 0
low_confidence_predictions = 0
predictions_total = 0

# nome modello base

BASEMODEL_NAME = "BaseModel"

# percorsi utili

BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
MODEL_PATH = Path(os.getenv("HF_MODEL_PATH", str(PROJECTDIR / "model")))

# recupero della pipeline per il modello scelto


def get_sentiment_pipeline(model_name=BASEMODEL_NAME):

    new_model_path = MODEL_PATH / model_name

    # controllo che sia già stato importato il modello

    global sentiment_pipeline, loaded_model_name

    if sentiment_pipeline is None:
        if model_name == BASEMODEL_NAME:
            if not new_model_path.exists():
                sentiment_pipeline = pipeline(
                    "sentiment-analysis", model=source_model_path
                )

                sentiment_pipeline.save_pretrained(new_model_path)
                loaded_model_name = BASEMODEL_NAME
            else:
                if not (new_model_path / "config.json").exists():
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis", model=source_model_path
                    )
                    sentiment_pipeline.save_pretrained(new_model_path)
                    loaded_model_name = BASEMODEL_NAME
                else:
                    sentiment_pipeline = load_saved_model(model_name)
                    loaded_model_name = BASEMODEL_NAME
        else:
            if not new_model_path.exists():
                sentiment_pipeline = pipeline(
                    "sentiment-analysis", model=source_model_path
                )
                print("MODELLO INESISTENTE. Caricato modello base")
                loaded_model_name = BASEMODEL_NAME
            else:
                if not (new_model_path / "config.json").exists():
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis", model=source_model_path
                    )
                    print("CARTELLA MODELLO VUOTA O CORROTTA. Caricato modello base")
                    loaded_model_name = BASEMODEL_NAME
                else:
                    sentiment_pipeline = load_saved_model(model_name)
                    loaded_model_name = model_name

    elif loaded_model_name != model_name:
        sentiment_pipeline = change_used_model(model_name)

    return sentiment_pipeline


# cambio del modello utilizzato


def change_used_model(model_name):
    global sentiment_pipeline, loaded_model_name

    if sentiment_pipeline is not None:
        if loaded_model_name != model_name:
            sentiment_pipeline = None
            sentiment_pipeline = get_sentiment_pipeline(model_name)
    else:
        sentiment_pipeline = get_sentiment_pipeline(model_name)

    return sentiment_pipeline


# funzione per fare le predizioni con il modello scelto


def make_prediction(text, batch_size=16, base_model=BASEMODEL_NAME, test=False):

    global \
        best_model_name, \
        sentiment_pipeline, \
        loaded_model_name, \
        low_confidence_predictions, \
        predictions_total

    if best_model_name is not None:
        sentiment_pipeline = get_sentiment_pipeline(best_model_name)
    else:
        best_model_choicer()

        if best_model_name is None:
            sentiment_pipeline = get_sentiment_pipeline(base_model)
        elif best_model_name != base_model:
            sentiment_pipeline = get_sentiment_pipeline(best_model_name)

    # text puo' essere una singola stringa oppure una lista di stringhe.
    # batch_size divide le liste grandi in blocchi piu' piccoli, cosi'
    # il modello non prova a processare troppi testi tutti insieme.

    predictions = sentiment_pipeline(
        text, batch_size=batch_size, truncation=True, max_length=512
    )

    # contiamo le volte che una predizione è stata poco sicura,
    # cioè con un punteggio inferiore a 0.6.

    if not test:
        for prediction in predictions:
            predictions_total += 1
            if prediction["score"] < 0.6:
                global low_confidence_predictions
                low_confidence_predictions += 1

        if check_confidence_score():
            if check_retraining_trigger() == "False":
                trigger_retraining_file_generator()
            
    elif test:
        pass

    return predictions, loaded_model_name


def save_model(model_version=BASEMODEL_NAME):
    """
    model_version = BASEMODEL_PATH -> standard
    Se model_version != da None il modello viene salvato nella
    cartella della versione iniziale
    """
    global loaded_model_name, sentiment_pipeline

    if model_version == BASEMODEL_NAME:
        sentiment_pipeline = get_sentiment_pipeline(model_version)
    else:
        if (loaded_model_name != BASEMODEL_NAME) and (
            loaded_model_name == model_version
        ):
            sentiment_pipeline.save_pretrained(MODEL_PATH / model_version)


def load_saved_model(model_version=None):
    """
    model_version = BASEMODEL_PATH -> standard
    Se model_version != da None il modello viene caricato dalla
    cartella della versione iniziale
    """

    if model_version is None:
        model_pipeline = pipeline(
            "sentiment-analysis",
            model=source_model_path,
        )
    else:
        saved_model_path = str(MODEL_PATH / model_version)

        model_pipeline = pipeline(
            "sentiment-analysis", model=saved_model_path, tokenizer=saved_model_path
        )

    return model_pipeline


# funzione per la tokenizzazione dei testi


def texts_tokenizer(texts, model_name=BASEMODEL_NAME):

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

    encoded = tokenizer(texts, truncation=True, max_length=512, padding=True)
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
        labels, predictions, average="macro", zero_division=0
    )
    # calcolo dell'accuracy

    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# funzioner per l'addestramento del modello


def train_model(
    dataset_name, new_version_name, model_trained=BASEMODEL_NAME, sample_size=None
):

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
        output_dir=str(MODEL_PATH / "training_arguments" / new_version_name),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        save_strategy="no",
        report_to="none",
    )

    # creazione del trainer

    trainer = Trainer(
        model=traning_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=arguments,
        compute_metrics=compute_metrics,
    )

    # valutazione del modello prima e dopo l'addestramento

    metrics["before training"] = trainer.evaluate(metric_key_prefix="before_training")
    # addestramento del modello

    trainer.train()

    # valutazione del modello dopo l'addestramento

    metrics["after training"] = trainer.evaluate(metric_key_prefix="after_training")

    trained_model = trainer.model

    # salvataggio del nuovo modello

    trainer.save_model(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    # controllo che il modello sia stato salvato correttamente

    if new_model_path.exists():
        if (new_model_path / "config.json").exists():
            print("modello salvato correttamente")
            csvfiles_manager.metrics_before_after(
                metrics, model_trained, new_version_name
            )
            global best_model_name
            best_model_choicer()

            print(f"Il modello migliore è: {best_model_name}")
        else:
            print("errore nel salvataggio del modello")
    else:
        print("errore nel salvataggio del modello")

    return trained_model, metrics


# def best_model_choicer() deve prendere in input le metriche di valutazione del modello e restituire il modello migliore da utilizzare per le predizioni.
# facciamo in modo che ci sia l'F1 score come metrica principale per la scelta del modello migliore. In caso di parità,
# si può considerare l'accuracy come metrica secondaria.


def best_model_choicer():
    """
    Questa funzione parte in automatico ogni volta che un modello viene addestrato con nuovi dati
    e confronta tutti i modelli presenti nel file della valutazioe e ne sceglie
    il migliore. In caso di parità, si può considerare l'accuracy come metrica secondaria.
    """

    global best_model_name, stop_bad_retraining

    # leggo il file delle metriche e lo ordino per F1 score e accuracy per
    # portare in cima il modello migliore

    try:
        metric_df = csvfiles_manager.read_file_metrics()
        if metric_df.empty:
            print("Non sono presenti metriche per scegliere il modello.")
            return None
        else:
            metric_df = metric_df.sort_values(by=["f1", "accuracy"], ascending=False)
            best_model_name = metric_df.iloc[0]["model_name"]

            global sentiment_pipeline
            sentiment_pipeline = change_used_model(best_model_name)
            print(f"Il modello migliore è: {best_model_name}")

            return best_model_name

    except FileNotFoundError:
        print(
            "Il file delle metriche non esiste. Non è possibile scegliere il modello migliore."
        )
        return None
    except pd.errors.EmptyDataError:
        print(
            "Il file delle metriche è vuoto. Non è possibile scegliere il modello migliore."
        )
        return None


def auto_retraining(current_model_name, dataset_name):
    """
    Questa funzione viene chiamata quando il modello corrente non è più affidabile
    e deve essere addestrato nuovamente con nuovi dati.
    """
    new_version_name = f"{current_model_name}_retrained"
    global stop_bad_retraining

    print(
        f"Avvio del retraining del modello {current_model_name} con il dataset {dataset_name}."
    )

    try:
        if stop_bad_retraining >= 3:
            print(
                "troppi retraining senza una miglioria. Necessario intervento manuale"
            )
            trigger_retraining_file_generator("stopped, manual intervention required")
        else:
            train_model(
                dataset_name, new_version_name, model_trained=current_model_name
            )
            global best_model_name

            if best_model_name == new_version_name:
                reset_retraining_trigger()
                stop_bad_retraining = 0
                print(
                    f"Retraining completato. Il nuovo modello è stato salvato come {new_version_name}."
                )
            else:
                print(
                    "retraining automatico concluso ma modello ancora non soddisfacente"
                )
                # scriviamo true nuovamente se il modello non è ancora performante
                trigger_retraining_file_generator()
                stop_bad_retraining += 1

    except Exception:
        trigger_retraining_file_generator()
        print("errore durante il retraining")
        raise  # rendiamo visibile l'errore nei log


def trigger_retraining_file_generator(trigger="True"):
    """
    Questa funzione crea un file di trigger per l'addestramento automatico del modello.
    """
    trigger_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trigger_file_path, "w") as f:
        f.write(trigger)


def reset_retraining_trigger():
    """
    Questa funzione resetta il file di trigger per l'addestramento automatico del modello.
    """
    global low_confidence_predictions, predictions_total
    low_confidence_predictions = 0
    predictions_total = 0

    if trigger_file_path.exists():
        with open(trigger_file_path, "w") as f:
            f.write("False")


def check_retraining_trigger():
    """
    Questa funzione controlla se il file di trigger per l'addestramento automatico del modello esiste e se contiene "True".
    Se il file non esiste, viene creato con il valore "False".
    """
    if not trigger_file_path.exists():
        trigger_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trigger_file_path, "w") as f:
            f.write("False")
        return False
    else:
        with open(trigger_file_path, "r") as f:
            content = f.read().strip()
            if content == "True":
                return content
            elif content == "False":
                return content
            elif content == "Running":
                return "run"


def check_confidence_score(threshold=0.2):
    """
    Questa funzione controlla se le predizioni del modello hanno un punteggio di confidenza
    inferiore a una soglia specificata. Se il punteggio di confidenza è inferiore alla soglia,
    viene restituito True, altrimenti False.
    """

    global low_confidence_predictions, predictions_total
    if predictions_total < 20:
        return False

    low_confidence_rate = low_confidence_predictions / predictions_total

    if low_confidence_rate > threshold:
        print(
            f"Il modello ha un punteggio di confidenza basso: {low_confidence_rate:.2%}."
        )
        return True
    else:
        return False
