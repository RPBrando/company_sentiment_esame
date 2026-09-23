from src.newdata_manager import Dataset
import pandas as pd

name_file = "train.csv"
fake_targets_general = pd.Series([
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",

        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral",

        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ])

def test_newdata_manager():

    data_obj = Dataset(name_file)

    x, y = data_obj.return_text_target()

    assert  len(x) == len(y)


def fake_return_text_target(sample_size = None):
    fake_text = [
    "I love this product",
    "The service was excellent",
    "This company is fantastic",
    "I am very satisfied",
    "The experience was amazing",

    "The product is acceptable",
    "The service was normal",
    "I have no particular opinion",
    "The experience was average",
    "It works as expected",

    "I hate this product",
    "The service was terrible",
    "This company is disappointing",
    "I am very dissatisfied",
    "The experience was awful"
]
    
    fake_targets = fake_targets_general

    return fake_text, fake_targets


def test_return_training_data(monkeypatch):
    dataset = Dataset("test")
    monkeypatch.setattr(dataset, "return_text_target", fake_return_text_target)
    monkeypatch.setattr(dataset,"target",fake_targets_general)

    text_train, text_test, target_train, target_test = dataset.return_training_data()


    assert all([text_train, text_test, target_train, target_test])
    assert len(target_train) / len(fake_targets_general) == 0.8
    assert len(text_train) == len(target_train)
    assert len(text_test) == len(target_test)
    assert set(target_train) <= {0,1,2}
