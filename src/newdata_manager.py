import pandas as pd
from pathlib import Path

# __file__ rappresenta il path di questo file Python.
# Partendo da qui costruiamo path stabili, che non dipendono dalla cartella
# da cui lanciamo il programma nel terminale.
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
        # Leggiamo il dataset grezzo

        raw_df = pd.read_csv(self.df_path, encoding=self.encoding)

        # Valutazione colonne necessarie
        required_columns = {"text", "sentiment"}

        # colonne richieste - colonne presenti = colonne mancanti.
        missing_columns = required_columns - set(raw_df.columns) # ricerca colonne mancanti

        if missing_columns:   # controllo colonne mancandi
            raise ValueError(f"Colonne mancanti nel dataset: {missing_columns}")

        # Alcune righe del dataset possono avere valori vuoti/NaN quindi puliamo il dataset prima di passarlo al modello
        raw_df = raw_df.dropna(subset=["text", "sentiment"]).copy()

        # Togliamo anche eventuali testi vuoti o composti solo da spazi
        raw_df = raw_df[raw_df["text"].astype(str).str.strip() != ""]

        # sample_size serve per fare test rapidi evitando di lanciare subito il modello su migliaia di testi.
        if sample_size is not None:
            raw_df = raw_df.head(sample_size)

        # Convertiamo le due colonne in liste di stringhe:
        # text andra' al modello, target andra' a classification_report.
        text = raw_df["text"].map(str).tolist()
        target = raw_df["sentiment"].map(str).tolist()
        return text, target
