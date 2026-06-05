import pandas as pd
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent
FILEPATH = BASEDIR/"data"
RESULTPATH = FILEPATH / "results.csv"
    
def read_file(): # funzione per leggere file - controllo dell'esistenza del file
    try:
        with open("results.csv", "r") as file:
            df = pd.read_csv(file)

    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["text", "label", "score"]
        )

    except pd.errors.EmptyDataError:
        df = pd.DataFrame(
                columns=["text", "label", "score"]
        )

    return df

def write_file(df): # scrittura del file
    
    with open("results.csv", "w") as file:
        df.to_csv(file, index = False)
        
    print(f"file scritto correttamente")