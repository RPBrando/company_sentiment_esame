from fastapi import FastAPI
from pydantic import BaseModel
import model

pipeline = model.Model.sentiment_pipeline

class Sentiment(BaseModel): # creazione del controllo del dato in ingresso
    sentiment: str

app = FastAPI()
