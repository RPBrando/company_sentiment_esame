import src.metrics as metric

prediction = ["positive", "negative"]
target = ["positive", "neutral"]


def test_metrics_creator():
    save = False
    met_dic = metric.metrics_creator(prediction, target, save)

    assert isinstance(met_dic, dict)


def test_metrics_creator_without_saving(monkeypatch):
    calls = []

    def fake_metrics_saver(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        metric.csvfiles_manager,
        "metrics_saver",
        fake_metrics_saver,
    )

    report = metric.metrics_creator(
        prediction=["positive", "neutral", "negative"],
        target=["positive", "neutral", "negative"],
        save=False,
        sample_size=3,
        model_name="test_model",
    )

    assert isinstance(report, dict)
    assert calls == []


def test_metrics_creator_with_saving(monkeypatch):
    calls = []

    def fake_metrics_saver(
        report, sample_size=None, model_name=None, dataset_name=None
    ):
        calls.append(
            {
                "report": report,
                "sample_size": sample_size,
                "model_name": model_name,
                "dataset_name": dataset_name,
            }
        )

    monkeypatch.setattr(
        metric.csvfiles_manager,
        "metrics_saver",
        fake_metrics_saver,
    )

    report = metric.metrics_creator(
        prediction=["positive", "neutral", "negative"],
        target=["positive", "neutral", "negative"],
        save=True,
        sample_size=3,
        model_name="test_model",
        dataset_name="test.csv",
    )

    assert isinstance(report, dict)
    assert len(calls) == 1
    assert calls[0]["report"] == report
    assert calls[0]["sample_size"] == 3
    assert calls[0]["model_name"] == "test_model"
    assert calls[0]["dataset_name"] == "test.csv"


def test_metrics_saver_csv(tmp_path, monkeypatch):
    import pandas as pd
    from src import csvfiles_manager

    # Reindirizziamo il salvataggio a un file temporaneo.
    test_file = tmp_path / "modelsummary.csv"

    monkeypatch.setattr(csvfiles_manager, "FILEPATH", tmp_path)
    monkeypatch.setattr(csvfiles_manager, "MODELCHECKFILE", test_file)

    report = {
        "negative": {
            "precision": 0.8,
            "recall": 0.7,
            "f1-score": 0.75,
            "support": 10,
        }
    }

    csvfiles_manager.metrics_saver(
        report,
        sample_size=3534,
        model_name="BaseModel",
        dataset_name="test.csv",
    )

    saved = pd.read_csv(test_file)

    assert len(saved) == 1
    assert saved.loc[0, "sample_size"] == 3534
    assert saved.loc[0, "model_name"] == "BaseModel"
    assert saved.loc[0, "dataset_name"] == "test.csv"
    assert saved.loc[0, "metric_class"] == "negative"
    assert saved.loc[0, "f1_score"] == 0.75
