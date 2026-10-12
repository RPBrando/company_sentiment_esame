from src import model
from pathlib import Path
import pytest

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

def fake_trigger_retraining_file_generator(trigger = True):
    return trigger

def fake_train_model():
    pass

def test_auto_retraining_success(monkeypatch, tmp_path):
    trigger_path = tmp_path / "trigger.txt"
    trigger_path.write_text("running")

    def fake_train(dataset_name, new_version_name, model_trained):
        model.best_model_name = new_version_name

    monkeypatch.setattr(model, "trigger_file_path", trigger_path)
    monkeypatch.setattr(model, "train_model", fake_train)
    monkeypatch.setattr(model, "stop_bad_retraining", 2)

    model.auto_retraining("BaseModel", "train.csv")

    assert trigger_path.read_text() == "False"
    assert model.stop_bad_retraining == 0

def test_auto_retraining_not_better(monkeypatch, tmp_path):
    trigger_path = tmp_path / "trigger.txt"

    def fake_train(dataset_name, new_version_name, model_trained):
        model.best_model_name = model_trained

    monkeypatch.setattr(model, "trigger_file_path", trigger_path)
    monkeypatch.setattr(model, "train_model", fake_train)
    monkeypatch.setattr(model, "stop_bad_retraining", 0)

    model.auto_retraining("BaseModel", "train.csv")

    assert trigger_path.read_text() == "True"
    assert model.stop_bad_retraining == 1

def test_auto_retraining_stopped(monkeypatch, tmp_path):
    trigger_path = tmp_path / "trigger.txt"

    def training_must_not_start(*args, **kwargs):
        raise AssertionError("Il training non doveva partire")

    monkeypatch.setattr(model, "trigger_file_path", trigger_path)
    monkeypatch.setattr(model, "train_model", training_must_not_start)
    monkeypatch.setattr(model, "stop_bad_retraining", 3)

    model.auto_retraining("BaseModel", "train.csv")

    assert trigger_path.read_text() == (
        "stopped, manual intervention required"
    )


def test_auto_retraining_error(monkeypatch, tmp_path):
    trigger_path = tmp_path / "trigger.txt"

    def fake_train(*args, **kwargs):
        raise RuntimeError("Training fallito")

    monkeypatch.setattr(model, "trigger_file_path", trigger_path)
    monkeypatch.setattr(model, "train_model", fake_train)
    monkeypatch.setattr(model, "stop_bad_retraining", 0)

    with pytest.raises(RuntimeError):
        model.auto_retraining("BaseModel", "train.csv")

    assert trigger_path.read_text() == "True"


def test_train_model(monkeypatch, tmp_path):

    # Dataset finto: evita di leggere il CSV reale.
    class FakeDataset:
        def __init__(self, dataset_name):
            self.dataset_name = dataset_name

        def return_training_data(self, sample_size=None):
            return (
                ["testo positivo", "testo negativo"],
                ["testo neutrale"],
                [2, 0],
                [1]
            )

    # Sostituisce Dataset.from_dict di Hugging Face.
    class FakeHFDataset:
        @staticmethod
        def from_dict(data):
            return data

    # Evita di scaricare il vero modello.
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_source):
            return "fake-model"

    # Serve per simulare il salvataggio del tokenizer.
    class FakeTokenizer:
        def save_pretrained(self, path):
            pass

    # Simula solamente le operazioni di Trainer usate dalla funzione.
    class FakeTrainer:
        trained = False
        evaluations = 0
        saved_path = None

        def __init__(
            self,
            model,
            train_dataset,
            eval_dataset,
            args,
            compute_metrics
        ):
            self.model = "fake-trained-model"

        def evaluate(self, metric_key_prefix):
            FakeTrainer.evaluations += 1

            return {
                f"{metric_key_prefix}_accuracy": 0.80,
                f"{metric_key_prefix}_f1": 0.80
            }

        def train(self):
            FakeTrainer.trained = True

        def save_model(self, path):
            FakeTrainer.saved_path = path

            # train_model controlla che config.json esista.
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").touch()

    # Simula la tokenizzazione.
    def fake_texts_tokenizer(texts, model_name):
        return {
            "input_ids": [[1, 2] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts]
        }

    # TrainingArguments deve solamente accettare i parametri.
    def fake_training_arguments(**kwargs):
        return kwargs

    # Sostituzione delle parti pesanti o che scrivono file reali.
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "models")
    monkeypatch.setattr(model.newdata_manager, "Dataset", FakeDataset)
    monkeypatch.setattr(model, "HFDataset", FakeHFDataset)
    monkeypatch.setattr(
        model,
        "AutoModelForSequenceClassification",
        FakeAutoModel
    )
    monkeypatch.setattr(model, "TrainingArguments", fake_training_arguments)
    monkeypatch.setattr(model, "Trainer", FakeTrainer)
    monkeypatch.setattr(model, "texts_tokenizer", fake_texts_tokenizer)
    monkeypatch.setattr(model, "tokenizer", FakeTokenizer())

    monkeypatch.setattr(
        model.csvfiles_manager,
        "metrics_before_after",
        lambda *args: None
    )

    monkeypatch.setattr(
        model,
        "best_model_choicer",
        lambda: "VersioneTest"
    )

    trained_model, metrics = model.train_model(
        dataset_name="train.csv",
        new_version_name="VersioneTest",
        model_trained=model.BASEMODEL_NAME,
        sample_size=100
    )

    assert trained_model == "fake-trained-model"
    assert FakeTrainer.trained is True
    assert FakeTrainer.evaluations == 2
    assert FakeTrainer.saved_path == (
        model.MODEL_PATH / "VersioneTest"
    )
    assert "before training" in metrics
    assert "after training" in metrics