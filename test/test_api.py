from src import api

text = {"sentiment": "text"}

def test_predict():

    result = api.predicton(text)


    assert isinstance(result, dict)

    assert "label" in result
    assert "score" in result

