from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import HTMLResponse

# Importiamo i moduli passando dal pacchetto src.
# In questo modo l'app funziona anche quando viene avviata da Docker con "src.api:app".
from src import schema
from src import model
from src import csvfiles_manager
from src import newdata_manager
from src import metrics
from src.monitoring import HTTPMetricsMiddleware, HTTPMonitoring


app = FastAPI()  # creazione API

app.state.http_monitoring = HTTPMonitoring()

app.add_middleware(
    HTTPMetricsMiddleware,
    monitoring=app.state.http_monitoring,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return app.state.http_monitoring.metrics_response()


@app.get("/", response_class=HTMLResponse)  # pagina iniziale
async def root():
    return """
            <!DOCTYPE html>
            <html lang="it">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="shortcut icon" href="#">
                <title>Company Sentiment</title>
                <style>
                    body {
                        margin: 0;
                        padding: 64px 24px;
                        background: #f8fafc;
                        color: #1e293b;
                        font-family: system-ui, sans-serif;
                        line-height: 1.7;
                    }
                    main { max-width: 640px; margin: 0 auto; }
                    h1 { margin-bottom: 8px; line-height: 1.2; }
                    .subtitle { margin-top: 0; color: #475569; }
                    nav { margin-top: 32px; }
                    nav a { display: block; margin: 12px 0; color: #1d4ed8; }
                    a:hover { text-decoration-thickness: 2px; }
                </style>
            </head>
            <body>
                <main>
                    <h1>Company Sentiment</h1>
                    <p class="subtitle">Progetto AI Engineering · Sentiment Analysis</p>
                    <p>
                        Servizio sviluppato con FastAPI e il modello RoBERTa di
                        Hugging Face per classificare testi in sentiment positivo,
                        neutro e negativo.
                    </p>
                    <p>
                        Il progetto integra test automatici, Docker, CI/CD e
                        strumenti di monitoraggio.
                    </p>
                    <nav aria-label="Collegamenti del progetto">
                        <a href="/docs">Documentazione interattiva — /docs</a>
                        <a href="/health">Stato API — /health</a>
                        <a href="https://github.com/RPBrando/company_sentiment_esame">
                            Repository GitHub
                        </a>
                    </nav>
                </main>
            </body>
            </html>
            """


# post. Ricezione del predict e scrittura su file


@app.post("/predict")
async def predicton(text: schema.Text, background: BackgroundTasks):
    testo = text.text

    # esecuzione della predizione del testo in input "grezza"

    prediction_raw, model_name = model.make_prediction(testo)

    # validazione del testo di input (** dizionario spacchettato)

    predict = schema.Prediction(**prediction_raw[0])

    # creazione del df con la sola stringa della predizione

    csvfiles_manager.prediction_saver(
        testo, predict.label, predict.score, model_name=model_name
    )
    # questa operazione fa partire in background il controllo del trigger senza
    # rallentare il risultato

    if model.check_retraining_trigger() == "True":
        # scriviamo running nel file di di trigger per non far partire un auto retraining
        # a un predizione ulteriore

        model.trigger_retraining_file_generator("Running")
        background.add_task(model.auto_retraining, model_name, "train.csv")
    elif model.check_retraining_trigger() == "run":
        print("Training già avviato")
        pass

    return predict


# creazione del file di sommario del dataset di prova


@app.post("/model-metrics")
async def dataset_prediction(
    # creazione di una query official che ci permette di non salvare le metriche
    # per piccoli test
    sample_size: int = Query(default=50, ge=1),
    official: bool = False,
):
    # Il test ufficiale è separato dal dataset di training.
    data = newdata_manager.Dataset("test.csv", encoding="utf-8")

    # La modalità ufficiale utilizza tutti i record.
    # La modalità rapida utilizza il campione richiesto.
    requested_size = None if official else sample_size
    texts, targets = data.return_text_target(sample_size=requested_size)

    prediction_raw, model_name = model.make_prediction(
        texts,
        batch_size=16,
        test=True,
    )

    predictions = [
        schema.Prediction(**single_prediction).label
        for single_prediction in prediction_raw
    ]

    metric_report = metrics.metrics_creator(
        predictions,
        targets,
        save=official,
        sample_size=len(targets),
        model_name=model_name,
        dataset_name="test.csv",
    )

    return metric_report


@app.get("/model-metrics")
async def showmetrics():
    df = csvfiles_manager.metrics_reader()
    return df.to_dict(orient="records")  # passaggio del df in dizionario json


@app.get("/results")  # stampa del file nel web
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

    uvicorn.run(app, "0.0.0.0", port=8000)
