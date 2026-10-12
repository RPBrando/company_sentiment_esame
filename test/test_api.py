from fastapi.testclient import TestClient
from pathlib import Path
from src import csvfiles_manager
from src import model
from src.api import app
from src import newdata_manager
from src import metrics

# test del client

fake_file = Path("tmp_results.csv")

# funzione fastapi per testare le API

client = TestClient(app)


def test_predict(monkeypatch):
    # Nei test non vogliamo caricare davvero il modello Hugging Face.
    # Creiamo quindi una risposta finta ma con la stessa forma della risposta reale.
    def fake_prediction(text):
        return [{"label": "neutral", "score": 0.90}], "Versione1"

    # Nei test non vogliamo scrivere davvero dentro data/results.csv.
    # Sostituiamo temporaneamente il salvataggio con una funzione vuota.
    def fake_prediction_saver(text, label, score, model_name=None):
        pass

# Uso del metodo monkeypatch di pytest per sostituire momentaneamente il modello con attributi creati in precedenza

    monkeypatch.setattr(model, "make_prediction", fake_prediction)
    monkeypatch.setattr(csvfiles_manager, "prediction_saver", fake_prediction_saver)

    response = client.post("/predict", json={"sentiment": "text"})

    assert response.status_code == 200

    result = response.json()
    assert result["label"] == "neutral"
    assert result["score"] == 0.90



def test_results_data(monkeypatch, tmp_path):

    fake_reader = csvfiles_manager.read_file

    prediction_saver = csvfiles_manager.prediction_saver
    


    fake_path = tmp_path / fake_file
    fake_path.touch()

    prediction_saver(
        "prova",
        "neutral",
        0.1,
        fake_path,
        model_name="Versione1"
        )

    def fake_reader_file ():
        return fake_reader(fake_path)

    monkeypatch.setattr(csvfiles_manager, "read_file", fake_reader_file)


    # controllo che l'api risponda

    response = client.get("/results")

    assert response.status_code == 200

    results = response.json()

    # controllo della struttura del file creato (NON VUOTO)

    assert "time" in results[0]
    assert "text" in results[0]
    assert "label" in results[0]
    assert "score" in results[0]
    assert results[0]["text"] == "prova"
    assert results[0]["label"] == "neutral"
    assert results[0]["score"] == 0.1


def test_results_no_data(monkeypatch, tmp_path):

    fake_path = tmp_path / fake_file
    fake_path.touch()

# salvo la funzione di lettura reale per evitare per poi sostituirla nel test

    file_reader = csvfiles_manager.read_file

# creazione della funzione per test con file vuoto

    def fake_csv_reader_file():
        return file_reader(fake_path)


# passiamo con monkeypatch la funzione con il file vuoto invece che quello reale

    monkeypatch.setattr(csvfiles_manager, "read_file", fake_csv_reader_file)
    responde = client.get("/results")
    assert responde.status_code == 200


    assert responde.json() == []

# summary test

def test_summary():

    response = client.get("/summary")
    result = response.json()

# lista del controllo dell chiavi del dizionario

    keys_list =["n. interaction", "sentiment count", "average score"]

    assert response.status_code == 200 
    assert isinstance(result, dict)
    assert list(result.keys()) == keys_list

def test_getmodel_metrics():

    keys_list = ["time","sample_size","metric_class","precision","recall","f1_score","support"]
    response = client.get("/model-metrics")
    result = response.json()

    assert response.status_code == 200
    assert isinstance(result[0], dict)
    assert keys_list == list(result[0].keys())

def test_postmodel_metrics(monkeypatch):

    metrics_creator = metrics.metrics_creator

# bisogna creare il file falso o inserire il save= False nel metodo
    def fake_new_data(self, sample_size = None):
        texts =["prova", "prova1", "prova2"]
        targets = ["positive", "neutral", "negative"]
        return texts, targets

    def fake_prediction(texts, batch_size = 16, test = False):
        return [
            {"label": "positive", "score": 0.90},
            {"label": "neutral", "score": 0.80},
            {"label": "negative", "score": 0.95}
        ], "Versione1"

    def fake_metrics_creator(predictions, targets, sample_size=None):
        return metrics_creator(
            predictions,
            targets,
            save=False,
            sample_size= sample_size
            )

    monkeypatch.setattr(newdata_manager.Dataset, "return_text_target", fake_new_data)
    monkeypatch.setattr(model, "make_prediction", fake_prediction)
    monkeypatch.setattr(metrics, "metrics_creator", fake_metrics_creator)

    response = client.post("/model-metrics")

    result = response.json()

    assert response.status_code == 200
    assert isinstance(result, dict)
    



    

    

