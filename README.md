---
title: ProfAI Company Sentiment Analysis
sdk: docker
app_port: 8000
---

# Company Sentiment Analysis

[![CI](https://github.com/RPBrando/company_sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/RPBrando/company_sentiment/actions/workflows/ci.yml)

Progetto MLOps per analizzare il sentiment di testi provenienti dai social media, conservare lo storico delle predizioni, valutare e riaddestrare il modello e monitorarne l'andamento nel tempo.

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
- confronto automatico delle versioni tramite F1-score e accuracy;
- attivazione del modello con le metriche migliori;
- trigger di retraining basato sulle predizioni a bassa confidenza;
- retraining in background tramite FastAPI;
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
test_model.ipynb           Smoke test reale del training
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

Il progetto conserva le metriche nel tempo per confrontare il modello base con le versioni addestrate. Dopo ogni training ordina le versioni per F1-score e, in caso di parita, per accuracy; la versione migliore viene caricata per le predizioni successive.

## Retraining automatico

Durante l'uso normale l'applicazione conta le predizioni con confidence inferiore a `0.6`. Dopo almeno 20 predizioni, se oltre il 20% risulta sotto questa soglia, viene attivato un file di trigger persistente.

Al termine di una richiesta verso `/predict` con trigger attivo, FastAPI:

1. imposta il trigger su `running` per evitare un secondo avvio immediato;
2. avvia il retraining in background sul dataset `train.csv`;
3. confronta la nuova versione con lo storico dei modelli;
4. disattiva il trigger se la nuova versione diventa la migliore;
5. riattiva il trigger se il training fallisce o non produce un modello migliore.

Dopo tre retraining consecutivi senza miglioramento il processo viene fermato e richiede un intervento manuale.

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

I test utilizzano fake e `monkeypatch` per verificare API, training e retraining senza scaricare o addestrare realmente il modello durante la CI.

### Smoke test del training

Il notebook [`test_model.ipynb`](test_model.ipynb) completa la verifica con un training reale su 300 righe. Usa cartelle isolate in `smoke_artifacts`, controlla il salvataggio di modello e tokenizer e verifica la registrazione delle metriche. Il numero di righe puo essere aumentato dopo la prima esecuzione riuscita.

Nell'ultima esecuzione locale lo smoke test ha usato 240 righe per il training e 60 per il test. L'F1 macro sul campione di test e passato da `0.680` a `0.872`. Il risultato dimostra il funzionamento del flusso, ma non sostituisce una valutazione finale su un campione piu ampio.

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

Sono implementati il servizio di predizione, la valutazione, il fine-tuning, il confronto e la selezione dei modelli, il retraining automatico, i test, la containerizzazione, il deploy, la persistenza e il monitoraggio.

Prima della consegna restano attivita di verifica e presentazione:

- scegliere la dimensione finale del campione di training in base alle risorse disponibili;
- eseguire in Google Colab il notebook di consegna e completarlo con spiegazione, risultati e link al repository.

## Limiti attuali

- Il retraining usa il dataset etichettato `train.csv`: l'applicazione non assegna automaticamente label corrette ai nuovi testi ricevuti dall'API.
- Il conteggio delle predizioni poco sicure e il numero dei tentativi falliti sono mantenuti in memoria e ripartono al riavvio del processo.
- Il training in background condivide CPU e memoria con l'API nello stesso container.
- La confidence del classificatore viene usata come segnale operativo, ma non dimostra da sola che una predizione sia corretta o errata.

## Scelte progettuali

- Il modello base viene recuperato da Hugging Face e salvato localmente al primo utilizzo.
- Ogni predizione registra il nome del modello che l'ha prodotta.
- I dati persistenti sono separati dall'immagine Docker.
- Un nuovo modello viene valutato e confrontato prima di sostituire quello attivo.
- Training e test utilizzano insiemi distinti per evitare data leakage.
