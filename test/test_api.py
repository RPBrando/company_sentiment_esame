from fastapi.testclient import TestClient
from src import csvfiles_manager
from src import model
from src.api import app

client = TestClient(app)


def test_predict(monkeypatch):
    # Nei test non vogliamo caricare davvero il modello Hugging Face.
    # Creiamo quindi una risposta finta ma con la stessa forma della risposta reale.
    def fake_prediction(text):
        return [{"label": "neutral", "score": 0.90}]

    # Nei test non vogliamo scrivere davvero dentro data/results.csv.
    # Sostituiamo temporaneamente il salvataggio con una funzione vuota.
    def fake_prediction_saver(text, label, score):
        pass

# Uso del metodo monkeypatch di pytest per sostituire momentaneamente il modello con attributi creati in precedenza

    monkeypatch.setattr(model, "make_prediction", fake_prediction)
    monkeypatch.setattr(csvfiles_manager, "prediction_saver", fake_prediction_saver)

    response = client.post("/predict", json={"sentiment": "text"})

    assert response.status_code == 200

    result = response.json()
    assert result["label"] == "neutral"
    assert result["score"] == 0.90
