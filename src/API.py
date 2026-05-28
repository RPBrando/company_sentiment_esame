from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model

pipeline = model.Model.sentiment_pipeline

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
        return """
            <!DOCTYPE html>
            <html lang="it">
            <link rel="shortcut icon" href="#">
            <head>
                <meta charset="UTF-8">
                <title>Progetto profAI</title>
            </head>
            <body>
                <div style="text-align: center; margin-top: 100px;">
                    <h1>Primo progetto reale di FastAPI!</h1>
                </div>
            </body>
            </html>
            """

@app.post("/prediction")
async def predicton(predict: schema.Prediction):
        with open("file.csv", "w") as file:
                data = file

        data.append(predict)
