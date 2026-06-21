import pandas as pd
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
DATAPATH = PROJECTDIR / "data"
FOLDER_DFPATH = DATAPATH / "sentiment_datasets"

class Dataset:

    """
    da passare:
        filename = Nome del file inserito in ../data/sentiment_datasets
        encoding = Tipo di caratteri per la codifica del dataset
    """
    def __init__(self, filename, encoding="latin1"):
        self.df_path = FOLDER_DFPATH / filename
        self.encoding = encoding
    
    def return_text_target(self, sample_size=None):
        raw_df = pd.read_csv(self.df_path, encoding=self.encoding)

        required_columns = {"text", "sentiment"}
        missing_columns = required_columns - set(raw_df.columns)
        if missing_columns:
            raise ValueError(f"Colonne mancanti nel dataset: {missing_columns}")

        if sample_size is not None:
            raw_df = raw_df.head(sample_size)

        text = raw_df["text"].astype(str).tolist()
        target = raw_df["sentiment"].astype(str).tolist()
        return text, target
