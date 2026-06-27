import pandas as pd
from pathlib import Path
from datetime import datetime

BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
FILEPATH = PROJECTDIR/"data"
RESULTPATH = FILEPATH / "results.csv"
MODELCHECKFILE = FILEPATH / "modelsummary.csv"

    
def read_file(file_path= RESULTPATH): # funzione per leggere file - controllo dell'esistenza del file
   
    try:
        with open(file_path, "r") as file:
            df = pd.read_csv(file)

    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["time","text", "label", "score"]
        )

    except pd.errors.EmptyDataError:
        df = pd.DataFrame(
                columns=["time","text", "label", "score"]
        )

    return df


def df_merge(df_predict):
     
     df_file = read_file()
     return pd.concat([df_file, df_predict], ignore_index= True)

def write_file(df): # scrittura del file
        FILEPATH.mkdir(parents = True, exist_ok=True)       # controllo sull'esistenza della cartella e sottocartelle
        with open(RESULTPATH, "w") as file:
            df.to_csv(file, index = False)
        
        print(f"file scritto correttamente")


def prediction_saver(text, label, score):
        time = datetime.now().strftime("%d-%m-%Y %H:%M:%S") # aggiunta del timestamp per andamento nel tempo
        
        df_prediction = pd.DataFrame([{     # creazione del df con la sola stringa della predizione
            "time": time,
            "text": text,
            "label": label,
            "score": score
        }])
        df_total = df_merge(df_prediction)  # creazione df con nuovo e già presenti in csv
        write_file(df_total)                # scrittura su file

# funzione per il ritorno del sommario

def df_summary(): 
    df = read_file()
    return {
         "n. interaction" : df.shape[0],
         "sentiment count" : df["label"].value_counts().to_dict(),
         "average score" : float(df["score"].mean())
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


def metrics_saver(metric_list, sample_size=None):
    FILEPATH.mkdir(parents=True, exist_ok=True)

    # classification_report(output_dict=True) restituisce un dizionario annidato:
    # ogni chiave principale e' una classe/metrica, ad esempio "negative",
    # "neutral", "positive", "accuracy", "macro avg".
    # Con transpose() trasformiamo quel dizionario in una tabella leggibile.
    metric_df = pd.DataFrame(metric_list).transpose()

    # Porto l'indice dentro una colonna normale.
    # Questo rende il CSV piu' facile da leggere e da usare in dashboard:
    # metric_class conterra' valori come negative, neutral, positive, accuracy.
    metric_df = metric_df.reset_index().rename(columns={
        "index": "metric_class",
        "f1-score": "f1_score"
    })

    # Aggiungo il timestamp del report.
    # Serve per confrontare nel tempo metriche vecchie e nuove su Grafana.
    metric_df.insert(0, "time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Salvo anche il numero di righe usate nel test.
    # Una accuracy su 50 esempi e una accuracy su 5000 esempi non hanno lo stesso peso.
    metric_df.insert(1, "sample_size", sample_size)

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
