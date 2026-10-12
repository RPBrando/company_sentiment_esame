import src.metrics as metric

prediction = ["positive", "negative"]
target = ["positive", "neutral"]


def test_metrics_creator():
    save = False
    met_dic = metric.metrics_creator(prediction, target, save)

    assert isinstance(met_dic, dict)
