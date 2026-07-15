from transformers import pipeline

model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = None


def get_sentiment_pipeline():

# controllo che sia già stato importato il modello

    global sentiment_pipeline

    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_path)

    return sentiment_pipeline

def make_prediction(text, batch_size=16):
    sentiment_pipeline = get_sentiment_pipeline()

    # text puo' essere una singola stringa oppure una lista di stringhe.
    # batch_size divide le liste grandi in blocchi piu' piccoli, cosi'
    # il modello non prova a processare troppi testi tutti insieme.
    return sentiment_pipeline(
        text,
        batch_size=batch_size,
        # Alcuni testi del dataset possono essere lunghi.
        # truncation=True li taglia alla lunghezza massima supportata dal modello.
        truncation=True,
        max_length=512
    )
