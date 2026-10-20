from fastapi.testclient import TestClient
from pathlib import Path
from src import csvfiles_manager
from src import model
from src.api import app
from src import newdata_manager
from src import metrics
import pytest

# test del client

fake_file = Path("tmp_results.csv")

# funzione fastapi per testare le API

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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

    response = client.post("/predict", json={"text": "text"})

    assert response.status_code == 200

    result = response.json()
    assert result["label"] == "neutral"
    assert result["score"] == 0.90


def test_results_data(monkeypatch, tmp_path):

    fake_reader = csvfiles_manager.read_file

    prediction_saver = csvfiles_manager.prediction_saver

    fake_path = tmp_path / fake_file
    fake_path.touch()

    prediction_saver("prova", "neutral", 0.1, fake_path, model_name="Versione1")

    def fake_reader_file():
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


def test_summary_empty(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        csvfiles_manager,
        "read_file",
        lambda: pd.DataFrame(columns=["label", "score"]),
    )

    response = client.get("/summary")

    assert response.status_code == 200
    assert response.json() == {
        "n. interaction": 0,
        "sentiment count": {},
        "average score": None,
    }


def test_summary_populated(monkeypatch):
    import pandas as pd

    fake_results = pd.DataFrame(
        {
            "label": ["positive", "negative", "positive"],
            "score": [0.9, 0.6, 0.9],
        }
    )

    monkeypatch.setattr(
        csvfiles_manager,
        "read_file",
        lambda: fake_results,
    )

    response = client.get("/summary")

    assert response.status_code == 200

    result = response.json()

    assert result["n. interaction"] == 3
    assert result["sentiment count"] == {
        "positive": 2,
        "negative": 1,
    }
    assert result["average score"] == pytest.approx(0.8)


def test_getmodel_metrics_empty(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        csvfiles_manager,
        "metrics_reader",
        lambda: pd.DataFrame(),
    )

    response = client.get("/model-metrics")

    assert response.status_code == 200
    assert response.json() == []


def test_getmodel_metrics_populated(monkeypatch):
    import pandas as pd

    fake_metrics = pd.DataFrame(
        [
            {
                "time": "2026-10-20 10:00:00",
                "sample_size": 3534,
                "model_name": "BaseModel",
                "dataset_name": "test.csv",
                "metric_class": "neutral",
                "precision": 0.7,
                "recall": 0.6,
                "f1_score": 0.65,
                "support": 1430,
            }
        ]
    )

    monkeypatch.setattr(
        csvfiles_manager,
        "metrics_reader",
        lambda: fake_metrics,
    )

    response = client.get("/model-metrics")

    assert response.status_code == 200

    result = response.json()
    assert len(result) == 1
    assert result[0]["model_name"] == "BaseModel"
    assert result[0]["dataset_name"] == "test.csv"
    assert result[0]["metric_class"] == "neutral"


def test_postmodel_metrics_test(monkeypatch):
    captured = {}

    def fake_new_data(self, sample_size=None):
        captured["requested_size"] = sample_size
        return (
            ["prova1", "prova2", "prova3"],
            ["positive", "neutral", "negative"],
        )

    def fake_prediction(texts, batch_size=16, test=False):
        captured["test_mode"] = test
        return [
            {"label": "positive", "score": 0.90},
            {"label": "neutral", "score": 0.80},
            {"label": "negative", "score": 0.95},
        ], "Versione1"

    def fake_metrics_creator(
        predictions,
        targets,
        save=True,
        sample_size=None,
        model_name=None,
        dataset_name=None,
    ):
        captured["save"] = save
        captured["sample_size"] = sample_size
        captured["model_name"] = model_name
        captured["dataset_name"] = dataset_name
        return {"macro avg": {"f1-score": 1.0}}

    monkeypatch.setattr(
        newdata_manager.Dataset,
        "return_text_target",
        fake_new_data,
    )
    monkeypatch.setattr(model, "make_prediction", fake_prediction)
    monkeypatch.setattr(metrics, "metrics_creator", fake_metrics_creator)

    response = client.post("/model-metrics")

    assert response.status_code == 200
    assert captured["requested_size"] == 50
    assert captured["test_mode"] is True
    assert captured["save"] is False
    assert captured["sample_size"] == 3
    assert captured["model_name"] == "Versione1"


def test_postmodel_metrics_official(monkeypatch):
    captured = {}

    def fake_new_data(self, sample_size=None):
        captured["requested_size"] = sample_size
        return (
            ["prova1", "prova2", "prova3"],
            ["positive", "neutral", "negative"],
        )

    def fake_prediction(texts, batch_size=16, test=False):
        return [
            {"label": "positive", "score": 0.90},
            {"label": "neutral", "score": 0.80},
            {"label": "negative", "score": 0.95},
        ], "Versione1"

    def fake_metrics_creator(
        predictions,
        targets,
        save=True,
        sample_size=None,
        model_name=None,
        dataset_name=None,
    ):
        captured["save"] = save
        captured["sample_size"] = sample_size
        captured["model_name"] = model_name
        captured["dataset_name"] = dataset_name
        return {"macro avg": {"f1-score": 1.0}}

    monkeypatch.setattr(
        newdata_manager.Dataset,
        "return_text_target",
        fake_new_data,
    )

    monkeypatch.setattr(model, "make_prediction", fake_prediction)
    monkeypatch.setattr(metrics, "metrics_creator", fake_metrics_creator)
    response = client.post("/model-metrics?official=true")

    assert response.status_code == 200
    assert captured["requested_size"] is None
    assert captured["save"] is True
    assert captured["sample_size"] == 3
    assert captured["model_name"] == "Versione1"
