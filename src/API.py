from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model

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

@app.post("/predict") # prova del post
async def predicton(text: schema.Sentiment):
        testo = text.sentiment
        predict = model.prediction(testo)
        print(predict)
        return predict



if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, "0.0.0.0", port= 8000)


        
