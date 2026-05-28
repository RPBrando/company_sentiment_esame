from fastapi import FastAPI
import model

pipeline = model.Model.sentiment_pipeline

app = FastAPI()

