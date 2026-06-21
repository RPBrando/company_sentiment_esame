from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model
import csvfiles_manager
import newdata_manager
import metrics


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
async def predicton(text: schema.Text):
        testo = text.sentiment
        prediction_raw = model.make_prediction(testo)     # esecuzione della predizione del testo in input "grezza"
        predict = schema.Prediction(**prediction_raw[0])   # validazione del testo di input (** dizionario spacchettato)

        csvfiles_manager.prediction_saver(        # creazione del df con la sola stringa della predizione
               testo,
               predict.label,
               predict.score
        )    
        
        return predict  

# creazione del file di sommario del dataset di prova  

@app.post("/test")
async def dataset_prediction(sample_size: int = 50):
    # Creo il gestore del dataset: lui sa dove si trova train.csv
    # e come estrarre testi e label reali.
    data = newdata_manager.Dataset("train.csv", "latin1")

    # texts contiene i testi da dare al modello.
    # targets contiene le label vere del dataset, usate per valutare il modello.
    texts, targets = data.return_text_target(sample_size=sample_size)

    # Il modello puo' ricevere una lista di testi e restituisce una lista
    # di dizionari, uno per ogni testo: {"label": ..., "score": ...}.
    prediction_raw = model.make_prediction(texts, batch_size=16)

    # Ogni output grezzo viene validato con Pydantic.
    # Alla fine teniamo solo la label predetta, per confrontarla con targets.
    predictions = [
        schema.Prediction(**single_prediction).label
        for single_prediction in prediction_raw
    ]

    # Confrontiamo label vere e label predette.
    # metrics_creator salva anche il report in data/modelsummary.csv.
    metric_report = metrics.metrics_creator(predictions, targets)
    return metric_report
    

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


        
