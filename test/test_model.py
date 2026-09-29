from src import model
from src import api


class FakePipeline:
    def __init__(self):
        self.saved_path = None

    def save_pretrained(self, path):
        self.saved_path = path


def test_model_pipeline(monkeypatch):
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(model, "sentiment_pipeline", fake_pipeline)
    monkeypatch.setattr(model, "loaded_model_name", "Versione1")

    model.save_model("Versione1")

    assert fake_pipeline.saved_path == model.MODEL_PATH / "Versione1"
