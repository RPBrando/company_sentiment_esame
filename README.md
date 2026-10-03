---
title: ProfAI Company Sentiment Analysis
sdk: docker
app_port: 8000
---

# Company Sentiment Analysis

[![CI](https://github.com/RPBrando/company_sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/RPBrando/company_sentiment/actions/workflows/ci.yml)

Progetto MLOps per analizzare il sentiment di testi provenienti dai social media, conservare lo storico delle predizioni, valutare le prestazioni del modello e monitorarne l'andamento nel tempo.

Il modello di partenza e [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), capace di classificare un testo come `positive`, `neutral` o `negative`.

## Funzionalita

- predizione del sentiment tramite API FastAPI;
- salvataggio di testo, label, confidence, timestamp e modello utilizzato;
- valutazione del modello su un dataset pubblico;
- calcolo di accuracy, precision, recall e F1-score;
- salvataggio dello storico delle metriche;
- gestione di modelli salvati e versionati;
- preparazione, tokenizzazione e suddivisione train/test dei dati;
- implementazione del fine-tuning tramite Hugging Face `Trainer`;
- test automatici con Pytest;
- build automatica dell'immagine Docker;
- deploy automatico su Hugging Face Spaces;
- persistenza dei dati tramite Storage Bucket;
- dashboard di monitoraggio su Grafana Cloud.

## Architettura

```text
src/
|-- api.py                 Endpoint FastAPI
|-- model.py               Caricamento, predizione e training del modello
|-- newdata_manager.py     Preparazione del dataset e split train/test
|-- metrics.py             Calcolo delle metriche
|-- csvfiles_manager.py    Lettura e scrittura degli storici CSV
`-- schema.py              Modelli Pydantic

test/                      Test automatici
.github/workflows/ci.yml   Pipeline CI/CD
Dockerfile                 Immagine dell'applicazione
```

## API

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `POST` | `/predict` | Analizza un testo e salva il risultato |
| `GET` | `/results` | Restituisce lo storico delle predizioni |
| `GET` | `/summary` | Restituisce il riepilogo dei sentiment |
| `POST` | `/model-metrics` | Valuta il modello su un campione del dataset |
| `GET` | `/model-metrics` | Restituisce lo storico delle metriche |
| `GET` | `/docs` | Documentazione OpenAPI interattiva |

Esempio di richiesta:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sentiment": "This company provides an excellent service"}'
```

## Dataset

Il dataset deve essere un file CSV contenente almeno queste colonne:

| Colonna | Contenuto |
|---|---|
| `text` | Testo da analizzare |
| `sentiment` | Label reale: positive, neutral o negative |

Prima del training vengono eseguiti:

1. rimozione dei valori mancanti e dei testi vuoti;
2. conversione delle label in valori numerici;
3. suddivisione stratificata in training set e test set;
4. tokenizzazione con il tokenizer associato al modello.

## Metriche

- **Accuracy**: percentuale complessiva di classificazioni corrette.
- **Precision**: affidabilita delle classificazioni prodotte dal modello.
- **Recall**: capacita di trovare gli elementi appartenenti a una classe.
- **F1-score**: media armonica di precision e recall.

Il progetto conserva le metriche nel tempo per confrontare il modello base con le versioni addestrate.

## Avvio locale

Il progetto utilizza Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api:app --reload
```

L'API sara disponibile su `http://127.0.0.1:8000` e la documentazione automatica su `http://127.0.0.1:8000/docs`.

## Test

```bash
python -m pytest test -q
```

I test utilizzano fake e `monkeypatch` per verificare API e logica applicativa senza scaricare o addestrare realmente il modello durante la CI.

## Docker

Build dell'immagine:

```bash
docker build -t company-sentiment .
```

Avvio del container:

```bash
docker run --rm -p 8000:8000 company-sentiment
```

## Persistenza

In locale i file vengono salvati nelle cartelle del progetto. Nel deploy i percorsi possono essere configurati tramite variabili d'ambiente:

| Variabile | Utilizzo |
|---|---|
| `PERSISTENT_DATA_PATH` | Storico delle predizioni e delle metriche |
| `HF_DATASET_PATH` | Dataset utilizzati per valutazione e training |
| `HF_MODEL_PATH` | Modelli salvati e versionati |

Su Hugging Face Spaces queste variabili puntano allo Storage Bucket montato, evitando la perdita dei dati dopo un riavvio del container.

## CI/CD

La pipeline GitHub Actions esegue in sequenza:

1. installazione delle dipendenze e test Pytest;
2. build dell'immagine Docker;
3. deploy su Hugging Face Spaces, soltanto dopo un push sul branch principale e se i passaggi precedenti terminano correttamente.

Applicazione online: [Hugging Face Space](https://the-rock93-profai-sa.hf.space/)

Documentazione API online: [Swagger UI](https://the-rock93-profai-sa.hf.space/docs)

## Monitoraggio

Grafana Cloud legge i dati esposti dall'API tramite il datasource Infinity. La dashboard mostra lo storico delle metriche del modello e la distribuzione del sentiment rilevato.

[Apri la dashboard pubblica](https://rpbrando.grafana.net/public-dashboards/0a02fb40a7ba4216b5adaa8f11983783)

## Stato del progetto

Sono completati il servizio di predizione, la valutazione, i test, la containerizzazione, il deploy, la persistenza e il monitoraggio.

La funzione di fine-tuning e gia implementata e comprende valutazione prima e dopo il training e salvataggio della nuova versione. Prima della consegna restano da completare:

- prova completa del training su un campione ridotto e successivamente sul dataset scelto;
- confronto dei modelli sullo stesso test set;
- regola di approvazione e attivazione del modello migliore;
- automazione del retraining;
- notebook Google Colab finale con spiegazione dei risultati e link al repository.

## Scelte progettuali

- Il modello base viene recuperato da Hugging Face e salvato localmente al primo utilizzo.
- Ogni predizione registra il nome del modello che l'ha prodotta.
- I dati persistenti sono separati dall'immagine Docker.
- Un nuovo modello deve essere valutato prima di sostituire quello attivo.
- Training e test utilizzano insiemi distinti per evitare data leakage.
