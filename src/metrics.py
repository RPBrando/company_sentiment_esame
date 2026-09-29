from sklearn.metrics import classification_report
from src import csvfiles_manager

def metrics_creator(prediction, target, save = True, sample_size=None, model_name=None):

    """
    prediction = -> predizione
    target -> dato reale
    save -> viene passato automaticamente come True. Il test invia questo parametro come False
    sample_size -> None | da passare per alleggerire il processo o dataset molto grandi
    """
    # classification_report confronta label vere e label predette.

    metric_report = classification_report(
        target,
        prediction,
        output_dict=True,
        zero_division=0
    )
    

    # Salvo il report su CSV insieme al sample_size usato per il test.
    if save == True:
        csvfiles_manager.metrics_saver(metric_report, sample_size=sample_size, model_name=model_name)
    return metric_report
