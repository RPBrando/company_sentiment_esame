import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

# __file__ rappresenta il path di questo file Python.
# Partendo da qui costruiamo path stabili, che non dipendono dalla cartella
# da cui lanciamo il programma nel terminale.
BASEDIR = Path(__file__).resolve().parent
PROJECTDIR = BASEDIR.parent
DATAPATH = PROJECTDIR / "data"
FOLDER_DFPATH = Path(
    os.getenv(
        "HF_DATASET_PATH",
        str(DATAPATH / "sentiment_datasets")
    )
)

class Dataset:

    """
    da passare:
        filename = Nome del file inserito in ../data/sentiment_datasets
        encoding = Tipo di caratteri per la codifica del dataset
    """
    def __init__(self, filename, encoding="latin1"):

        self.df_path = FOLDER_DFPATH / filename
        self.encoding = encoding
        self.text = None
        self.target = None
    
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
        self.text = raw_df["text"].map(str).tolist()
        self.target = raw_df["sentiment"].map(str)

        target_list = self.target.to_list()


        return self.text, target_list

    def return_training_data(self):
        text, _ = self.return_text_target()
            
        label_encoder = {
        "positive": 2,
        "neutral": 1,
        "negative": 0
        }

        target_encoded = self.target.map(label_encoder).to_list()
        text_train, text_test, target_train, target_test = train_test_split(
            text, 
            target_encoded, 
            test_size= .2, 
            random_state= 0, 
            shuffle= True,
            stratify=target_encoded
            )

        return text_train, text_test, target_train, target_test

