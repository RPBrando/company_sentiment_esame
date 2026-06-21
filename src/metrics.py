from sklearn.metrics import classification_report
import csvfiles_manager

def metrics_creator(prediction, target):
    metric_report = classification_report(
        target,
        prediction,
        output_dict=True,
        zero_division=0
    )
    csvfiles_manager.metrics_saver(metric_report)
    return metric_report
