from transformers import pipeline

model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = pipeline("sentiment", model= model_path)
