import pandas as pd
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent #path generale del progetto
DATAPATH = BASEDIR / "data"
FOLDER_DFPATH = DATAPATH / "sentiment_datasets"

class Dataset:

    """
    da passare:
        filename = Nome del file inserito in ../data/sentiment_datasets
        econding = Tipo di caretteti per la codifica del dataset
    """
    def __init__(self, filename, encoding):
        self.DFPATH = FOLDER_DFPATH / filename
        self.encoder = encoding
    
    def return_text_target(self):
        raw_df = pd.read_csv(self.DFPATH, encoding=self.encoder)

        raw_df["sentiment"].value_counts()

        text = raw_df["text"].values
        target = raw_df["sentiment"].values
        return text, target

