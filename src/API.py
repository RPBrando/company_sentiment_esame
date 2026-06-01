from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model

pipeline = model.Model.sentiment_pipeline # Import del modello nella pipeline

app = FastAPI() # creazione API

@app.get("/", response_class=HTMLResponse) # prova del get
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

@app.post("/prediction") # prova del post
async def predicton(predict: schema.Prediction):
        with open("file.csv", "w") as file:
                data = file

        data.append(predict)


if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, "0.0.0.0", port= 8000)


        
