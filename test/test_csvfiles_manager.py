import src.csvfiles_manager as csv
import pandas as pd


# pimo test sul sommario del csv

def test_df_summary():

    summary = csv.df_summary()

# controllo del tipo di istanza del metodo df_summary

    assert isinstance(summary, dict)

# controllo della presenza delle seguenti chiavi del dizionario

    assert "n. interaction" in summary
    assert "sentiment count" in summary
    assert "average score" in summary

# controllo che la lettura delle metriche ritorni un Dataset Pandas

def test_metrics_reader():
    metrics = csv.metrics_reader()
    assert isinstance(metrics, pd.DataFrame)

# controllo che la lettura del file ritorni un Dataset Pandas

def test_read_file():

    df = csv.read_file()
    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "text" in df.columns
    assert "label" in df.columns
    assert "score" in df.columns
    assert "model_name" in df.columns
    

