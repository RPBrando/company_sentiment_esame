from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model
import csvfiles_manager


app = FastAPI() # creazione API

@app.get("/", response_class=HTMLResponse) # pagina iniziale
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

@app.post("/predict")                       # post. Ricezione del predict e scrittura su file
async def predicton(text: schema.Sentiment):
        testo = text.sentiment              # estrazione del testo dalla classe Sentiment in schema
        predict = model.make_prediction(testo)   # recupero della predizione del modello

        csvfiles_manager.prediction_saver(        # creazione del df con la sola stringa della predizione
               testo,
               predict[0]["label"],
               predict[0]["score"]
        )    
        
        return predict                     

@app.get("/results") # stampa del file nel web
async def showing_prediction():
    df = csvfiles_manager.read_file()
    return df.to_dict(orient="records")

# da completare

@app.get("/summary")
async def showing_summary():
       return csvfiles_manager.df_summary()


# partenza del file con uvicorn

if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, "0.0.0.0", port= 8000)


        
