from pydantic import BaseModel

class Sentiment(BaseModel): # creazione del controllo del dato in ingresso
    sentiment: str

class Prediction(BaseModel): # creazione della crasse che rappresenta il dato in arrivo dalla predizione
    
    prediction: str
    score: float

