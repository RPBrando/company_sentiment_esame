import pandas as pd
from pathlib import Path
from datetime import datetime
import os

DATA_FORMATTING = "%Y-%m-%d %H:%M:%S"
BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
FILEPATH = Path(  # necessario per trasformare stringa in percorso
    os.getenv(
        "PERSISTENT_DATA_PATH",  # variabile su Huggingface per i dati persistenti
        str(PROJECTDIR / "data"),  # valore di default (quando si lavora in locale)
    )
)

RESULTPATH = FILEPATH / "results.csv"
MODELCHECKFILE = FILEPATH / "modelsummary.csv"
MODELTRAININGFILE_COMPARE = FILEPATH / "modeltrainingcomparison.csv"


# LETTURA DEL FILE


def read_file(
    file_path=RESULTPATH,
):  # funzione per leggere file - controllo dell'esistenza del file

    try:
        with open(file_path, "r") as file:
            df = pd.read_csv(file)
            if "model_name" not in df.columns:
                df["model_name"] = "unknown"
            else:
                df["model_name"] = df["model_name"].fillna("unknown")

    except FileNotFoundError:
        df = pd.DataFrame(columns=["time", "text", "label", "score", "model_name"])
        write_file(df, file_path)

    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["time", "text", "label", "score", "model_name"])
        write_file(df, file_path)

    return df


# funzione per la lettura del file di metriche, con controllo dell'esistenza del
# file e gestione dei casi di file vuoto o inesistente.


def read_file_metrics(
    file_path=MODELTRAININGFILE_COMPARE,
):  # funzione per leggere file - controllo dell'esistenza del file

    try:
        with open(file_path, "r") as file:
            df = pd.read_csv(file)
            if "model_name" not in df.columns:
                df["model_name"] = "unknown"
            else:
                df["model_name"] = df["model_name"].fillna("unknown")

    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["model_name", "loss", "accuracy", "precision", "recall", "f1"]
        )
        write_file(df, file_path)

    except pd.errors.EmptyDataError:
        df = pd.DataFrame(
            columns=["model_name", "loss", "accuracy", "precision", "recall", "f1"]
        )
        write_file(df, file_path)

    return df


# UNIONE DEL DF PRESENTE SUL FILE E DI QUELLO PREDETTO


def df_merge(df_predict, result_path=RESULTPATH):

    df_file = read_file(result_path)
    return pd.concat([df_file, df_predict], ignore_index=True)


# SCRITTURA DEL FILE


def write_file(df, result_path=RESULTPATH):  # scrittura del file
    FILEPATH.mkdir(
        parents=True, exist_ok=True
    )  # controllo sull'esistenza della cartella e sottocartelle
    with open(result_path, "w") as file:
        df.to_csv(file, index=False)

    print("file scritto correttamente")


# SALVATAGGIO DELLE PREDIZIONI


def prediction_saver(
    text, label, score, result_path=RESULTPATH, model_name=None
):  # funzione per salvare le predizioni in un file csv

    time = datetime.now().strftime(
        DATA_FORMATTING
    )  # aggiunta del timestamp per andamento nel tempo

    df_prediction = pd.DataFrame(
        [
            {  # creazione del df con la sola stringa della predizione
                "time": time,
                "text": text,
                "label": label,
                "score": score,
                "model_name": model_name,
            }
        ]
    )
    df_total = df_merge(
        df_prediction, result_path
    )  # creazione df con nuovo e già presenti in csv
    write_file(df_total, result_path)  # scrittura su file


# funzione per il ritorno del sommario di tutte le predizioni effettuate e salvate


def df_summary():
    df = read_file()
    return {
        "n. interaction": int(df.shape[0]),
        "sentiment count": df["label"].value_counts().to_dict(),
        "average score": float(df["score"].mean()),
    }


# lettura del file di metriche


def metrics_reader():

    # lettura del file
    try:
        with open(MODELCHECKFILE, "r") as metrics_file:
            read_results = pd.read_csv(metrics_file)
    # in caso di file vuoto
    except pd.errors.EmptyDataError:
        read_results = pd.DataFrame()
        print(read_results)
    # in caso di file inesistente
    except FileNotFoundError:
        with open(MODELCHECKFILE, "w") as metrics_file:
            read_results = pd.DataFrame()
            pass

    # il tipo di risultato inviato è sempre di tipo Dataframe Pandas anche se vuoto per non far rompere l'API

    return read_results


def metrics_saver(metric_list, sample_size=None, model_name=None):
    """
    metrics.metric_report(output_dict=True) restituisce un dizionario annidato
    ogni chiave principale e' una classe/metrica, ad esempio "negative","neutral", "positive", "accuracy", "macro avg".
    Con transpose() trasformiamo quel dizionario in una tabella leggibile.
    """
    # controllo dell'esistenza della path giusta

    FILEPATH.mkdir(parents=True, exist_ok=True)

    # creazione del df con le metriche

    metric_df = pd.DataFrame(metric_list).transpose()

    # Porto l'indice dentro una colonna normale per facilità di lettura e comodità in dashboard

    metric_df = metric_df.reset_index().rename(
        columns={"index": "metric_class", "f1-score": "f1_score"}
    )

    # Aggiungo il timestamp del report (utile per grafana sopratutto)
    metric_df.insert(0, "time", datetime.now().strftime(DATA_FORMATTING))

    # Salvo anche il numero di righe usate nel test.

    metric_df.insert(1, "sample_size", sample_size)

    # salvo il nome del modello usato per il test. Se non viene passato, salvo il nome del modello caricato di default.

    # Aggiungo il nome del modello al dataframe
    if model_name is not None:
        metric_df.insert(2, "model_name", model_name)

    try:
        # Se esiste gia' uno storico metriche, lo leggo e aggiungo il nuovo report.

        old_metric_df = pd.read_csv(MODELCHECKFILE)
        metric_df = pd.concat([old_metric_df, metric_df], ignore_index=True)
    except FileNotFoundError:
        # Se il file non esiste ancora, il report corrente sara' il primo storico.
        pass
    except pd.errors.EmptyDataError:
        # Se il file esiste ma e' vuoto, ripartiamo dal report corrente.
        pass

    metric_df.to_csv(MODELCHECKFILE, index=False)


def metrics_before_after(metrics, model_training_name, model_trained_name):
    """ "
    Questa funzione prende il risultato delle metrice prima e dopo dell'addestramento
    e lo salva in un file csv per poterlo visualizzare in grafana.
    """

    # creazione del dizionario per salvare i dati da scrivere nel file csv

    compare_data = {}

    # creazione del dataframe con le metriche prima e dopo l'addestramento

    compare_data[model_training_name] = metrics["before training"]
    compare_data[model_trained_name] = metrics["after training"]

    # funzione interna per rimuovere la chiave dalle metriche

    def remove_key(keyToRemove, dictToClean):
        """
        Funzione per rimuovere una chiave da un dizionario annidato.
        """

        tmp_dict = {}

        # ciclo per rimuovere la chiave dal dizionario annidato

        for key, value in dictToClean.items():
            key = key.removeprefix(keyToRemove)
            tmp_dict[key] = value

        return tmp_dict

    # standardizzazione dei nomi delle metriche per uniformità e leggibilità nel file di
    # confronto tra le metriche prima e dopo l'addestramento

    compare_data[model_training_name] = remove_key(
        "before_training_", compare_data[model_training_name]
    )  # metriche pre training
    compare_data[model_trained_name] = remove_key(
        "after_training_", compare_data[model_trained_name]
    )  # metriche post training

    # creazione del dataframe con le metriche prima e dopo l'addestramento

    compare_df = pd.DataFrame(compare_data).transpose()

    # Porto l'indice dentro una colonna normale per facilità di lettura e comodità in dashboard

    compare_df = compare_df.reset_index().rename(columns={"index": "model_name"})

    # scrittura del file csv con le metriche prima e dopo l'addestramento

    try:
        # Se esiste gia' uno storico metriche, lo leggo e aggiungo il nuovo report.

        old_metric_df = pd.read_csv(MODELTRAININGFILE_COMPARE)
        compare_df = pd.concat([old_metric_df, compare_df], ignore_index=True)

    except FileNotFoundError:
        # Se il file non esiste ancora, il report corrente sara' il primo storico.
        pass
    except pd.errors.EmptyDataError:
        # Se il file esiste ma e' vuoto, ripartiamo dal report corrente.
        pass

    write_file(compare_df, MODELTRAININGFILE_COMPARE)
