from transformers import pipeline

model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = pipeline("sentiment-analysis", model= model_path)

def make_prediction(text, batch_size=16):
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
