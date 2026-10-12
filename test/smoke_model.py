from src import model


def test_real_model_inference(monkeypatch):
    expected_labels = {"positive", "neutral", "negative"}

    # Forza l'utilizzo del modello base ed evita la selezione di versioni
    # eventualmente presenti nello storico delle metriche.
    monkeypatch.setattr(model, "best_model_name", model.BASEMODEL_NAME)

    # Forza un caricamento reale, indipendente dallo stato lasciato
    # da eventuali test eseguiti precedentemente nello stesso processo.
    monkeypatch.setattr(model, "sentiment_pipeline", None)
    monkeypatch.setattr(model, "loaded_model_name", None)

    predictions, model_name = model.make_prediction(
        "This product is excellent and I am very satisfied.", test=True
    )

    assert model_name == model.BASEMODEL_NAME
    assert isinstance(predictions, list)
    assert len(predictions) == 1

    prediction = predictions[0]

    assert {"label", "score"} <= set(prediction)
    assert prediction["label"].lower() in expected_labels
    assert isinstance(prediction["score"], float)
    assert 0.0 <= prediction["score"] <= 1.0

    configured_labels = {
        str(label).lower()
        for label in model.sentiment_pipeline.model.config.id2label.values()
    }

    assert configured_labels == expected_labels
