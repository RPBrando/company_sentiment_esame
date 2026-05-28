from pydantic import BaseModel

class Sentiment(BaseModel): # creazione del controllo del dato in ingresso
    sentiment: str

class Prediction(BaseModel):
    prediction: str
    score: float

