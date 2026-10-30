---
title: ProfAI Company Sentiment Analysis
sdk: docker
app_port: 8000
---

# Company Sentiment

Progetto AI Engineering per classificare testi in sentiment **positivo, neutro e negativo**, conservare le predizioni e applicare pratiche MLOps di test, distribuzione, valutazione e monitoraggio.

Il servizio usa FastAPI e il modello pre-addestrato [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). I testi vengono inviati dal chiamante: non è implementata una raccolta automatica dai social.

[![CI](https://github.com/RPBrando/company_sentiment_esame/actions/workflows/ci.yml/badge.svg)](https://github.com/RPBrando/company_sentiment_esame/actions/workflows/ci.yml)

- [Repository](https://github.com/RPBrando/company_sentiment_esame)
- [Notebook di consegna](company_sentiment_consegna.ipynb)
- [Apri il notebook in Google Colab](https://colab.research.google.com/github/RPBrando/company_sentiment_esame/blob/main/company_sentiment_consegna.ipynb)

Il collegamento Colab diventa disponibile dopo la pubblicazione del notebook sul branch `main`.

## Funzionalità e stato effettivo

| Componente | Implementazione |
|---|---|
| Inferenza | Pipeline Hugging Face, caricamento differito e riuso del modello in memoria |
| API | Predizione, storico, riepilogo, valutazione, health check e metriche HTTP |
| Persistenza | CSV per predizioni, valutazioni ufficiali e confronti di training |
| Fine-tuning | `train_model()` con Hugging Face Trainer; una epoca, valutazione prima/dopo e salvataggio |
| Selezione | Ordinamento dello storico di training per F1 macro e poi accuracy |
| Retraining | Trigger sulla bassa confidence e task in background avviato da `/predict` |
| CI/CD | Ruff, pytest, build Docker e sincronizzazione condizionata verso Hugging Face Spaces |
| Monitoraggio | Prometheus per HTTP; Grafana con datasource Prometheus e Infinity |

**Il workflow attuale non esegue un training reale.** Il training è disponibile nel codice e nel notebook storico `test_model.ipynb`; i test CI del training usano fake. Il requisito della traccia che richiede il training automatico nella CI/CD non è quindi ancora coperto dal workflow presente.

## Struttura e flusso

| File o directory | Responsabilità |
|---|---|
| [src/api.py](src/api.py) | Homepage, endpoint FastAPI e avvio del retraining in background |
| [src/model.py](src/model.py) | Modello/tokenizer, inferenza, training, selezione e trigger |
| [src/newdata_manager.py](src/newdata_manager.py) | Lettura CSV, pulizia, encoding delle label e split |
| [src/metrics.py](src/metrics.py) | Report di classificazione con scikit-learn |
| [src/csvfiles_manager.py](src/csvfiles_manager.py) | Lettura/scrittura degli storici e riepilogo |
| [src/schema.py](src/schema.py) | Schemi degli input e delle predizioni |
| [src/monitoring.py](src/monitoring.py) | Counter, Histogram e middleware HTTP |
| [test/](test/) | Test automatici e smoke test separato dell'inferenza |
| [monitoring/](monitoring/) | Scraping, datasource, dashboard e alert provisionati |
| [compose.yaml](compose.yaml) | Stack locale API, Prometheus e Grafana |
| [company_sentiment_consegna.ipynb](company_sentiment_consegna.ipynb) | Dimostrazione riproducibile attraverso le funzioni del progetto |
| [test_model.ipynb](test_model.ipynb) | Smoke test storico del training, con output locali conservati |

Una richiesta `POST /predict` passa da `schema.Text` a `make_prediction()`. Il modello produce label e score; `prediction_saver()` registra il risultato. L'API verifica poi il trigger e restituisce la predizione. Il modello base viene scaricato e salvato al primo utilizzo, se non è già disponibile. Le versioni locali vengono caricate con il relativo tokenizer.

## Ambiente e avvio locale

La CI e il Dockerfile usano **Python 3.11**. Le dipendenze dirette sono elencate in [requirements.txt](requirements.txt). Il primo utilizzo reale del modello richiede accesso a Hugging Face e risorse per scaricare e caricare i pesi.

Dalla radice del repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Homepage: `http://127.0.0.1:8000/`. Documentazione interattiva: `http://127.0.0.1:8000/docs`.

## API

| Metodo | Endpoint | Comportamento |
|---|---|---|
| GET | `/` | Homepage del progetto |
| GET | `/health` | Restituisce `{"status": "ok"}`; non controlla pesi o dataset |
| POST | `/predict` | Classifica un testo, salva il risultato e controlla il trigger |
| GET | `/results` | Restituisce tutte le predizioni salvate |
| GET | `/summary` | Conteggio cumulativo delle interazioni e delle classi, confidence media |
| POST | `/model-metrics` | Valutazione su `test.csv`, rapida oppure ufficiale |
| GET | `/model-metrics` | Legge lo storico delle valutazioni ufficiali |
| GET | `/metrics` | Metriche HTTP in formato Prometheus |
| GET | `/docs` | Swagger UI generata da FastAPI |

Il campo di input è **`text`**:

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This company provides an excellent service"}'
```

La risposta contiene `label` e `score`. Il nome del modello viene registrato nel CSV. `score` è la confidence del classificatore e non garantisce la correttezza della predizione. I testi vengono troncati a 512 token; lo schema non esclude stringhe vuote o composte da spazi.

`src/client.py` è uno script dimostrativo legacy che invia ancora il campo `sentiment`: usare l'esempio sopra. `src/client_test.py` richiede una valutazione rapida su un massimo di 10.000 righe, senza salvarla nello storico. Entrambi inviano richieste quando eseguiti o importati.

## Dataset

La documentazione originaria identifica la fonte come Kaggle *Tweet Sentiment Extraction*. Il CSV di training incluso contiene anche colonne aggiuntive: il codice utilizza esclusivamente `text` e `sentiment`. Il repository non contiene uno script di acquisizione o trasformazione che ricostruisca la provenienza esatta dei file.

| File | Impiego | Encoding | Distribuzione nella versione esaminata |
|---|---|---|---|
| `data/sentiment_datasets/train.csv` | Training e validation interna | Latin-1 | Tracciato in Git |
| `data/sentiment_datasets/test.csv` | Holdout della valutazione ufficiale | UTF-8 | Disponibile nel workspace locale, ma escluso da Git |

**Un clone pulito non contiene `test.csv`.** Prima di usare `POST /model-metrics`, collocare l'holdout etichettato nella directory dei dataset. L'endpoint applicativo non gestisce questa assenza e fallisce. Il notebook di consegna presenta i risultati disponibili senza eseguire valutazioni.

Entrambi i file devono avere `text` e `sentiment`, con label `negative`, `neutral`, `positive`. Per riprodurre la baseline storica serve l'esatto holdout, non un CSV omonimo con testi o label diversi.

Controllo in sola lettura dei file locali del 30 ottobre 2026:

| Dataset | Record grezzi | Record con testo/label non vuoti | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| Training | 27.481 | 27.480 | 7.781 | 11.117 | 8.582 |
| Holdout | 3.534 | 3.534 | 1.001 | 1.430 | 1.103 |

SHA-256 dei file verificati:

```text
train.csv  c56362dadb70153967782dccf817dd21bdc4fa004fedbdae9d32a52ac81bda9a
test.csv   4d1fc2995ce88eb7d435f18563e93361f35956c3e74ed22572b9614eff6d04e6
```

Non risultano `textID` condivisi o testi esattamente identici tra i due file. Normalizzando spazi e maiuscole, 26 righe del test hanno un equivalente nel training: è un limite di indipendenza del test set. Non è verificata l'assenza di sovrapposizione con il pretraining del modello esterno.

## Fine-tuning e selezione del modello

`train_model(dataset_name, new_version_name, model_trained=BASEMODEL_NAME, sample_size=None)` riutilizza il modello pre-addestrato. La procedura:

1. elimina valori mancanti e testi vuoti;
2. se richiesto, prende le prime `sample_size` righe valide;
3. codifica negative → 0, neutral → 1, positive → 2;
4. divide `train.csv` in training 80% e validation 20%, con stratificazione e `random_state=0`;
5. tokenizza con padding e troncamento a 512 token;
6. valuta il modello iniziale, esegue una epoca e rivaluta sulla stessa validation;
7. salva pesi, tokenizer e confronto delle metriche;
8. seleziona e carica il modello con F1 macro maggiore, usando accuracy come secondo criterio.

Parametri impostati nel codice: batch training/evaluation 8, learning rate `2e-5`, una epoca, nessun checkpoint intermedio e nessun reporting esterno. Con `sample_size=300` vengono usati **240 testi per il training e 60 per la validation**. I nomi `X_test` e `test_dataset` interni al training indicano questa validation, non il file ufficiale `test.csv`.

`compute_metrics()` calcola accuracy, precision macro, recall macro e F1 macro. Il Trainer restituisce anche loss e statistiche operative. Il confronto è scritto in `modeltrainingcomparison.csv` e la funzione restituisce modello addestrato e valutazioni prima/dopo.

La selezione usa lo storico della validation, non `modelsummary.csv`. Non controlla che confronti storici diversi usino gli stessi dati; non impone una valutazione sull'holdout prima dell'attivazione. Gli esperimenti vanno eseguiti con percorsi dedicati per evitare la contaminazione con storici precedenti.

## Valutazione ufficiale e storici

Con `test.csv` disponibile:

```bash
# Prime 50 righe valide; nessun salvataggio del report
curl -X POST 'http://127.0.0.1:8000/model-metrics?sample_size=50'

# Tutte le righe valide; salvataggio nello storico
curl -X POST 'http://127.0.0.1:8000/model-metrics?official=true'
```

La modalità ufficiale ignora `sample_size`. Entrambe usano il modello selezionato dall'applicazione e `make_prediction(..., test=True)`: le predizioni di valutazione non alimentano il trigger di retraining. La modalità rapida usa `head`, non un campionamento casuale.

`classification_report(..., zero_division=0)` produce precision, recall, F1 e support per classe, accuracy, macro average e weighted average.

| File | Contenuto |
|---|---|
| `results.csv` | Timestamp, testo, label, score e nome del modello |
| `modelsummary.csv` | Report ufficiali con timestamp, modello, dataset e numerosità |
| `modeltrainingcomparison.csv` | Metriche prima/dopo il training usate dalla selezione |

Nella riga `accuracy` di `modelsummary.csv`, lo scalare viene replicato nelle colonne numeriche dalla conversione in DataFrame: il relativo `support` non è una numerosità. Usare `sample_size` e i support delle classi. I timestamp applicativi sono locali e privi di timezone.

### Evidenze dei risultati

Il notebook storico [test_model.ipynb](test_model.ipynb) conserva gli output di uno smoke training locale: 300 righe complessive, validation di 60 esempi, F1 macro da **0,680098** a **0,871674**. Il kernel registrato è Python 3.13.9. È un'evidenza di esecuzione del flusso, non una stima finale della qualità sull'holdout. Le sue celle di preparazione puntano al repository originale; il notebook di consegna ne riassume i risultati senza eseguire quel codice.

La documentazione precedente riportava una baseline del 20 ottobre 2026, modello `BaseModel` su 3.534 esempi:

| Metrica | Valore storico riportato |
|---|---:|
| Accuracy | 0,709960 |
| Macro F1 | 0,709542 |
| Weighted F1 | 0,699702 |
| F1 negative | 0,741149 |
| F1 neutral | 0,613053 |
| F1 positive | 0,774424 |
| Recall neutral | 0,509091 |

Quel report era descritto come salvato nel volume Docker. **Il `modelsummary.csv` tracciato è vuoto:** questi valori sono riferimenti documentali, non risultati ricalcolati nel controllo finale. Il notebook di consegna li riporta esplicitamente come risultati storici documentati.

## Retraining automatico

Nelle predizioni ordinarie il codice conta gli score strettamente inferiori a `0.6`. Con almeno 20 predizioni e una quota a bassa confidence strettamente superiore al 20%, può attivare il trigger. I contatori sono cumulativi nel processo, non una finestra mobile.

Il file è `PERSISTENT_DATA_PATH/retrain_triggered/trigger.txt`:

| Contenuto | Gestione |
|---|---|
| `False` | Il controllo della confidence può scrivere `True` |
| `True` | `/predict` scrive `Running` e programma `auto_retraining()` |
| `Running` | Il lettore restituisce `run`: una richiesta successiva non lo riattiva nel normale percorso sequenziale |
| `stopped, manual intervention required` | Stato scritto quando il contatore dei training senza miglioramento ha raggiunto tre |

Il retraining usa `train.csv`, non le predizioni recenti prive di label. Se il candidato diventa il migliore, resetta trigger e contatori; altrimenti riattiva il trigger e incrementa il contatore dei training senza miglioramento. Le eccezioni riattivano il trigger ma non incrementano quel contatore. Il limite viene controllato all'invocazione successiva.

Non c'è uno scheduler periodico: l'avvio dipende dalle richieste a `/predict`. Il file non è un lock atomico tra processi. Se il file manca, il lettore lo crea con `False` ma restituisce inizialmente il booleano `False`, mentre i controlli confrontano stringhe: l'eventuale attivazione viene rinviata a una lettura successiva. I contatori si azzerano al riavvio; i nomi dei candidati possono essere riutilizzati.

## Test e CI/CD

Dalla radice del progetto:

```bash
python -m ruff check src test
python -m ruff format --check src test
python -m pytest test -q
```

I 25 test raccolti normalmente verificano API, lettura/scrittura CSV, report, split e flussi di training/retraining simulati. `monkeypatch` sostituisce le operazioni pesanti; `tmp_path` isola parte delle scritture. Alcuni test usano i percorsi predefiniti: eseguirli con variabili d'ambiente dedicate se sono presenti dati operativi.

[test/smoke_model.py](test/smoke_model.py) contiene una prova reale dell'inferenza, esclusa dalla raccolta predefinita per il nome del file. Si può richiamare esplicitamente con `python -m pytest test/smoke_model.py -q`; può scaricare e salvare il modello.

Il [workflow](.github/workflows/ci.yml) esegue `test` → `docker-build` → `deploy`. Ruff e pytest precedono la build; il deploy parte solo su push a `main`. Le pull request verso `main`/`master` e gli altri push eseguono i controlli previsti senza deploy. Il training reale e il notebook non sono eseguiti dalla CI.

La build controlla il Dockerfile; l'immagine resta sul runner. Il deploy usa `huggingface/hub-sync@v0.3.0` verso `The-rock93/profai_SA` e richiede il secret `HF_TOKEN`. Sincronizza il repository, non promuove automaticamente i modelli addestrati in un esperimento locale.

Controllo locale del 30 ottobre 2026, codice al commit `293c2b43a4445dcd01d3c44e41f1185378e21c1e`: **25 test superati**, Ruff lint e formatting superati, Python 3.13.9 e dipendenze corrispondenti a `requirements.txt`. Test eseguiti con directory temporanee e Hugging Face offline. È comparso un avviso di deprecazione dell'integrazione TestClient/httpx. Non sono stati eseguiti training, inferenza reale, build o deploy durante questo controllo; non equivale a una nuova esecuzione GitHub Actions su Python 3.11.

## Docker, persistenza e monitoraggio

Per la sola API:

```bash
docker build -t company-sentiment .
docker run --rm -p 127.0.0.1:8000:8000 company-sentiment
```

Questo avvio non configura volumi persistenti. Il Dockerfile usa Python 3.11 slim e installa Torch CPU senza fissarne la versione, poi [requirements_docker.txt](requirements_docker.txt). Le altre dipendenze dirette hanno versioni fissate; non è presente un lock completo delle dipendenze transitive.

Per lo stack locale completo, impostare `GRAFANA_ADMIN_PASSWORD` nel terminale, quindi:

```bash
docker compose up --build -d
```

| Servizio | URL locale |
|---|---|
| API | `http://127.0.0.1:8000` |
| Grafana | `http://127.0.0.1:3000` — utente `admin`, password configurata |
| Prometheus | `http://127.0.0.1:9090` |

Compose monta volumi separati per dati API, modelli, Grafana e Prometheus. La directory dei dataset è un bind mount in sola lettura. I CSV nel volume API non aggiornano automaticamente quelli nel checkout. `docker compose down` arresta lo stack; l'opzione `-v` eliminerebbe anche i volumi.

| Variabile | Default locale | Uso |
|---|---|---|
| `PERSISTENT_DATA_PATH` | `data/` | Storici e file di trigger |
| `HF_MODEL_PATH` | `model/` | Modelli e output del Trainer |
| `HF_DATASET_PATH` | `data/sentiment_datasets/` | Dataset di input |
| `HF_HOME` | Default della libreria Hugging Face | Cache; Compose la colloca nel volume dei modelli |

Le variabili vanno impostate **prima dell'import** dei moduli. Su un servizio remoto la persistenza richiede uno storage effettivamente montato: le sole variabili non creano uno Storage Bucket.

### Monitoraggio HTTP

Il middleware espone un Counter di richieste per endpoint/metodo/status e un Histogram delle durate. Esclude `/metrics`; misura fino all'invio del corpo della risposta, senza attendere il completamento del training in background. La durata comprende più del solo tempo d'inferenza.

Prometheus raccoglie `/metrics` ogni 15 secondi, con timeout di 5 secondi e retention locale di sette giorni. La dashboard HTTP mostra disponibilità dello scraping, richieste al secondo, errori 5xx al secondo e latenza p95 di `/predict`. I tassi su cinque minuti escludono `/health`; la p95 include anche richieste non riuscite alla route.

L'alert `FastAPI scraping unavailable` controlla `up < 1` ogni 30 secondi, con pending period di un minuto. Gli stati senza dati e di errore sono `NoData` ed `Error`. Non sono provisionati contact point o notifiche esterne.

### Monitoraggio del modello e del sentiment

Infinity legge `/results`, `/summary` e `GET /model-metrics`. I 15 pannelli mostrano interazioni, distribuzione delle classi, tabella delle metriche, F1 per classe, accuracy, precision/recall, sample size e confidence. Il refresh è di due ore; il range iniziale è di 30 giorni. Il refresh legge i report già salvati e non avvia una valutazione.

`/summary` è cumulativo; i pannelli non separano automaticamente modelli o dataset. Il confronto interno `modeltrainingcomparison.csv` non è esposto da un endpoint dedicato. Le metriche ML non sono esportate in Prometheus. I formati timestamp dichiarati in diversi pannelli usano `/`, mentre il CSV usa `-`: il rendering effettivo va controllato nell'istanza Grafana.

Il progetto non implementa rilevazione automatica di data drift, prediction drift o concept drift, né alert su F1/recall. Confidence e classi predette sono segnali osservabili; non dimostrano da soli degrado o drift. Ripetere la valutazione sul medesimo holdout non misura concept drift.

### Riferimenti remoti

La configurazione e la documentazione indicano lo [Space Hugging Face](https://the-rock93-profai-sa.hf.space/), la relativa [Swagger UI](https://the-rock93-profai-sa.hf.space/docs) e una [dashboard Grafana Cloud](https://rpbrando.grafana.net/public-dashboards/0a02fb40a7ba4216b5adaa8f11983783). Disponibilità, storage e configurazione delle istanze remote non sono stati verificati nel controllo locale.

