from sklearn.metrics import classification_report
import csvfiles_manager

def metrics_creator(prediction, target, sample_size=None):
    # classification_report confronta label vere e label predette.
    # target = label reali del dataset
    # prediction = label generate dal modello
    metric_report = classification_report(
        target,
        prediction,
        output_dict=True,
        zero_division=0
    )

    # Salvo il report su CSV insieme al sample_size usato per il test.
    csvfiles_manager.metrics_saver(metric_report, sample_size=sample_size)
    return metric_report
