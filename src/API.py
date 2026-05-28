from fastapi import FastAPI
import schema
import model

pipeline = model.Model.sentiment_pipeline

app = FastAPI()

