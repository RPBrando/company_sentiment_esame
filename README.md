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
- preparazione, tokenizzazione e suddivisione training/validation dei dati;
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
|-- newdata_manager.py     Preparazione del dataset e split training/validation
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
| `POST` | `/model-metrics` | Valutazione rapida sul test set o ufficiale con `official=true` |
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
3. suddivisione stratificata di `train.csv` in training (80%) e validation (20%), con `random_state=0`;
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

Grafana Cloud legge i dati esposti dall'API tramite il datasource Infinity. La dashboard mostra lo storico delle metriche del modello e la distribuzione del sentiment rilevato. Il repository comprende inoltre il provisioning dello stack locale Prometheus/Grafana in [compose.yaml](compose.yaml) e nella cartella [monitoring](monitoring/).

[Apri la dashboard pubblica](https://rpbrando.grafana.net/public-dashboards/0a02fb40a7ba4216b5adaa8f11983783)

### Monitoraggio tecnico e monitoraggio ML

I due livelli hanno sorgenti e finalità distinte:

| Livello | Sorgente | Indicatori |
|---|---|---|
| Tecnico | Metriche HTTP FastAPI esposte da `/metrics` e raccolte da Prometheus | Disponibilità dello scraping, request rate, tasso di errori 5xx e latenza HTTP p95 di `/predict` |
| ML | CSV persistenti esposti come JSON da `/results`, `/summary` e `GET /model-metrics` | Interazioni, distribuzione delle classi predette, metriche di valutazione e confidence |

Il [middleware HTTP](src/monitoring.py) misura la durata fino all'invio del corpo della risposta, non soltanto il tempo di inferenza. La [configurazione Prometheus](monitoring/prometheus/prometheus.yml) prevede scraping ogni 15 secondi e timeout di 5 secondi. La [dashboard tecnica](monitoring/grafana/dashboards/company-sentiment-http-monitoring.json) contiene quattro pannelli: request rate e tasso di errori 5xx sono calcolati su finestre di 5 minuti ed escludono `/health`; `/metrics` è escluso dalla strumentazione. Il pannello 5xx rappresenta errori al secondo, non un conteggio assoluto.

L'alert provisionato [FastAPI scraping unavailable](monitoring/grafana/provisioning/alerting/api-scraping.yaml) valuta `up{job="company-sentiment-api"} < 1` ogni 30 secondi, con pending period di 1 minuto. I casi senza dati e di errore di esecuzione hanno rispettivamente stato `NoData` ed `Error`. La verifica manuale riportata per il progetto ha osservato le transizioni Pending, Firing e Normal mediante simulazione della condizione negativa: non equivale alla prova di un guasto reale dello scraping. Il repository documenta la regola, ma non contiene evidenze eseguibili di questa simulazione né provisioning di contact point o notifiche esterne; il recapito effettivo dipende dalla configurazione dell'istanza Grafana e non è verificabile dai soli file.

La [dashboard ML Infinity](monitoring/grafana/dashboards/company-sentiment-monitor.json) contiene 15 pannelli: due per interazioni e distribuzione dei sentiment, una tabella delle metriche, uno storico F1 per classe, uno storico accuracy, sei pannelli precision/recall, uno per sample size e tre per confidence. Macro F1 e weighted F1 sono disponibili nelle righe della tabella, senza pannelli dedicati. Il range predefinito è `now-30d` → `now`, con refresh ogni `2h`. Il riepilogo `/summary` resta cumulativo: selezionare 30 giorni nella dashboard non trasforma la distribuzione dei sentiment in una statistica limitata a quei giorni. I pannelli non separano automaticamente le valutazioni per modello o dataset: la comparabilità va verificata prima di interpretare gli andamenti.

Le metriche ML restano nei CSV e non vengono duplicate in Prometheus. La configurazione locale è verificabile nel repository; lo stato della dashboard Cloud e le impostazioni dell'istanza in esecuzione richiedono una verifica distinta.

### Protocollo di valutazione e tracciabilità

Il protocollo utilizza il dataset pubblico Kaggle *Tweet Sentiment Extraction*, con i file locali in [data/sentiment_datasets](data/sentiment_datasets/):

| Dataset | Utilizzo | Codifica nel loader |
|---|---|---|
| `train.csv` | Training e validation interna stratificata 80/20 | Latin-1 |
| `test.csv` | Holdout ufficiale: 3.534 testi etichettati `negative`, `neutral`, `positive` | UTF-8 |

Lo split interno di `train.csv` è chiamato `test` in alcune variabili del codice, ma è utilizzato come `eval_dataset` dal Trainer e non coincide con l'holdout ufficiale. Il flusso di retraining attivato da `/predict` passa `train.csv`. La funzione generica di training accetta un nome di dataset: l'esclusione di `test.csv` dal training è quindi anche un vincolo operativo da rispettare, non un divieto imposto dal loader.

L'[endpoint di valutazione](src/api.py) distingue:

| Modalità | Selezione dopo la pulizia | Persistenza delle metriche |
|---|---|---|
| Rapida | Prime 50 righe valide per default; `sample_size` configurabile | `save=False` |
| Ufficiale: `POST /model-metrics?official=true` | Tutte le righe valide di `test.csv`; `sample_size` ignorato | `save=True` |

Il [loader](src/newdata_manager.py) elimina valori mancanti di testo/label e testi vuoti. Il file verificato contiene 3.534 record validi; il codice registra il numero effettivamente valutato, senza imporre quel numero con un controllo automatico. La selezione rapida usa `head`, quindi è riproducibile a file e ordine invariati, ma non è un campione casuale o stratificato e non produce una baseline ufficiale.

Entrambe le modalità chiamano `make_prediction(..., batch_size=16, test=True)` in [src/model.py](src/model.py), usando il modello selezionato dall'applicazione, con troncamento a 512 token. Non impongono permanentemente `BaseModel`. `test=True` esclude queste predizioni dai contatori di bassa confidence e dalla generazione del relativo trigger. `save=False` impedisce il salvataggio delle metriche, ma non garantisce l'assenza di altri effetti del caricamento/selezione del modello, come il salvataggio iniziale del modello base.

[src/metrics.py](src/metrics.py) calcola `classification_report` con `zero_division=0`. [src/csvfiles_manager.py](src/csvfiles_manager.py) salva in `modelsummary.csv` timestamp `time`, `model_name` restituito dall'inferenza, `dataset_name`, `sample_size` effettiva e le colonne `metric_class`, `precision`, `recall`, `f1_score`, `support`. Il report contiene accuracy, metriche per classe, macro average e weighted average. Nella riga `accuracy` il valore scalare viene replicato nelle colonne numeriche dalla conversione in DataFrame: il suo campo `support` non è una numerosità; utilizzare `sample_size` e i support delle singole classi. Il timestamp è locale, senza indicazione del fuso orario.

Il nome del modello e quello del dataset consentono una tracciabilità minima, ma non identificano immutabilmente file sovrascritti. Per confronti omogenei mantenere invariati test set, etichette, pulizia, impostazioni d'inferenza e definizioni delle metriche, conservando versioni distinte dei modelli. Le righe legacy prive di metadati non permettono di ricostruire automaticamente tale provenienza.

SHA-256 documentale di `test.csv`, ricalcolato in sola lettura durante la verifica della documentazione:

```text
4d1fc2995ce88eb7d435f18563e93361f35956c3e74ed22572b9614eff6d04e6
```

Non è presente un controllo automatico del checksum ad ogni valutazione. Il test set deve rimanere immutato. Il confronto locale con `train.csv` non rileva `textID` condivisi né testi esattamente identici; normalizzando spazi e maiuscole con `split`/`join` e `casefold`, 26 righe del test hanno equivalenti nel training (circa 0,74%). Si documenta questo limite senza alterare lo split ufficiale. L'indipendenza rispetto a eventuali dati di pretraining del modello esterno non è dimostrabile dai file locali.

### Baseline ufficiale e riferimento storico

La valutazione ufficiale del **20 ottobre 2026**, riportata nella verifica manuale del progetto e visualizzata in Grafana, utilizza `BaseModel` (`cardiffnlp/twitter-roberta-base-sentiment-latest`) su tutti i 3.534 record:

| Metrica | Risultato riportato |
|---|---:|
| Accuracy | 0.709960 |
| Macro F1 | 0.709542 |
| Weighted F1 | 0.699702 |
| F1 negative | 0.741149 |
| F1 neutral | 0.613053 |
| F1 positive | 0.774424 |
| Recall neutral | 0.509091 |

La baseline ufficiale è stata ottenuta tramite `POST /model-metrics?official=true` sull'API eseguita in Docker Compose, con risposta HTTP 200. Il report è stato verificato direttamente nel file `/app/data/modelsummary.csv` del container: 3.534 esempi, modello `BaseModel`, dataset `test.csv` e sei righe di metriche.

Il file `data/modelsummary.csv` nel workspace locale è separato dal volume persistente Docker e non viene aggiornato automaticamente dalle richieste al container.

Il recall della classe neutral, pari a circa il 50,9%, motiva il monitoraggio specifico delle sue prestazioni.

### Data drift: segnali disponibili e metodo proposto

Il data drift è un cambiamento nella distribuzione degli input rispetto a un riferimento. Lunghezza dei testi, lingua, vocabolario, argomenti, hashtag e abbreviazioni sono caratteristiche da osservare. Le variazioni delle classi predette sono invece segnali di *prediction drift*: da sole, come le variazioni della confidence, non dimostrano un cambiamento degli input o un peggioramento dell'accuratezza.

Oggi sono disponibili testi, timestamp, classi predette e score in `results.csv`, con distribuzione cumulativa e grafici di confidence in dashboard. Non sono implementate statistiche degli input per finestre temporali né rilevazione automatica del drift.

Come evoluzione minima si propone di confrontare finestre omogenee di produzione con un riferimento, iniziando dalla lunghezza dei testi. Un possibile metodo è il Population Stability Index (PSI), con intervalli di discretizzazione fissati sul riferimento e una gestione esplicita delle frequenze nulle.

| PSI | Criterio operativo proposto |
|---|---|
| < 0,10 | Nessun allarme secondo questa euristica; non prova l'assenza di drift |
| Da 0,10 a 0,20 inclusi | Warning e analisi del cambiamento |
| > 0,20 | Investigazione approfondita |

Le soglie non sono validate statisticamente sul progetto. Numerosità, variabilità naturale, scelta degli intervalli e cambiamenti di modello possono influenzare l'interpretazione. Il PSI sulla lunghezza non rileva necessariamente cambiamenti semantici; nessun calcolo PSI è attualmente automatizzato.

### Concept drift: verifica con dati etichettati

Il concept drift è un cambiamento nella relazione tra testo e sentiment corretto, ad esempio per espressioni che cambiano significato nel tempo o nel contesto. Le predizioni di produzione non contengono ground truth e non bastano a dimostrarlo.

La procedura proposta, non automatizzata, è:

1. raccogliere un campione rappresentativo di testi recenti;
2. ottenere etichette affidabili, preferibilmente umane, con criteri coerenti;
3. valutare il modello attivo e confrontare precision, recall e F1 con un riferimento etichettato metodologicamente compatibile;
4. analizzare errori e cambiamenti degli input, distinguendoli da modifiche delle regole di annotazione.

Un calo delle prestazioni segnala una possibile model degradation, ma non identifica automaticamente il concept drift come causa. Bias del campione, errori di annotazione e scarsa numerosità per classe limitano le conclusioni.

### Model degradation e criteri di intervento proposti

La model degradation è una riduzione delle prestazioni rispetto a un riferimento appropriato. Si propongono due indicatori principali: **macro F1**, che assegna uguale peso alle tre classi, e **recall neutral**, per evidenziare peggioramenti specifici che una media può nascondere. Accuracy, weighted F1, metriche delle singole classi e support restano disponibili per l'analisi.

| Indicatore | Warning: calo assoluto | Critico: calo assoluto | Warning sul test fisso | Critico sul test fisso |
|---|---:|---:|---:|---:|
| Macro F1 | ≥ 0,03 | ≥ 0,05 | ≤ 0,679542 | ≤ 0,659542 |
| Recall neutral | ≥ 0,05 | ≥ 0,10 | ≤ 0,459091 | ≤ 0,409091 |

Le soglie numeriche derivano dai valori arrotondati della baseline del 20 ottobre: sono criteri operativi iniziali di investigazione, non soglie statisticamente validate né alert implementati. La condizione critica ha precedenza sul warning. Sono applicabili a confronti sul medesimo holdout e con la stessa procedura; per nuovi campioni di produzione occorre un riferimento appropriato, considerando composizione per classe e incertezza statistica. Una futura validazione potrebbe stimare la variabilità delle differenze, ad esempio con bootstrap appaiato sul test fisso.

In caso di superamento delle soglie o anomalie, la risposta proposta è verificare integrità dei dati e protocollo, identificare modello/dataset/numerosità, analizzare gli errori per classe e i cambiamenti degli input, quindi raccogliere etichette recenti quando necessario. Soltanto dopo questa analisi si valuta l'opportunità del retraining e si confronta il candidato con il modello corrente su dati omogenei. Un allarme non giustifica da solo il rilascio di un nuovo modello.

### Collegamento con il retraining e limiti attuali

Il [retraining automatico](#retraining-automatico) già implementato usa una regola diversa dalle soglie ML proposte: score strettamente inferiore a `0.6`, almeno 20 predizioni e quota a bassa confidence strettamente superiore al 20%. I contatori sono cumulativi nel processo e ripartono al riavvio; non rappresentano una finestra temporale mobile. La confidence è un segnale euristico, senza una procedura di calibrazione presente nel codice, e non dimostra drift o accuratezza insufficiente.

La selezione automatica attuale ordina F1 e accuracy in `modeltrainingcomparison.csv`, prodotto dalle valutazioni interne al training. Non utilizza la baseline ufficiale di `modelsummary.csv` e non verifica automaticamente l'omogeneità dei dataset delle valutazioni storiche. La verifica controllata del candidato sull'holdout è dunque una procedura operativa proposta, non un blocco già imposto al rilascio automatico.

Restano da realizzare statistiche degli input su finestre temporali, raccolta periodica di ground truth recente e verifiche statistiche del degrado. Non sono configurati alert ML automatici per data drift, concept drift, F1 o recall. Il file locale delle predizioni non contiene attualmente uno storico popolato sufficiente per calibrare soglie; la quantità di dati nei volumi di esecuzione va verificata separatamente.

L'orchestrazione periodica tramite Airflow, con eventuali retry, è una fase successiva e non è implementata nel flusso documentato. Le priorità future sono raccogliere evidenze etichettate, confrontare modelli su dati omogenei e valutare i segnali temporali già disponibili, senza introdurre una piattaforma MLOps aggiuntiva. L'obiettivo è un protocollo comprensibile, riproducibile e mantenibile, adeguato alla traccia d'esame.

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
- Training e validation derivano da uno split di `train.csv`; l'holdout ufficiale è `test.csv`, con i limiti di indipendenza documentati nella sezione monitoraggio.
