import pandas as pd
from pathlib import Path
from datetime import datetime

BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
FILEPATH = PROJECTDIR/"data"
RESULTPATH = FILEPATH / "results.csv"
    
def read_file(): # funzione per leggere file - controllo dell'esistenza del file
    try:
        with open(RESULTPATH, "r") as file:
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


def df_summary(): # funzione per il ritorno del sommario
    df = read_file()
    return {
         "n. interaction" : df.shape[0],
         "sentiment count" : df["label"].value_counts().to_dict(),
         "average score" : float(df["score"].mean())
    }
