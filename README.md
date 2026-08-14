# company_sentiment

[![CI](https://github.com/RPBrando/company_sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/RPBrando/company_sentiment/actions/workflows/ci.yml)

## Obiettivo del progetto

Controllare l'andamento del sentiment aziendale online. Creare pipeline di CI/CD in modo 
da garantire qualità in tutte le fasi di sviluppo e deployment.
Controllare in modo costante, con implementazione di DB esterni:
- andamento delle performance del modello utilizzato
- valutazione della necessità di eventuale apprendimento

Creazione futura di metodi di auto-apprendimento.

## Funzionalità principali
- Analisi sentiment su testo singolo
- Salvataggio storico predizioni
- Lettura dataset pubblico
- Valutazione modello
- Salvataggio metriche nel tempo

## Architettura del progetto
- api.py:
    Contenitore di tutti gli endpoint presenti nel progetto.
    Qui sono presenti metodi per leggere/scrivere metriche e analisi sentiment effettuate.
- model.py:
    Contenitore del modello e del metodo di predizione.
- schema.py:
    Contenitore delle classi Pydantic per il controllo della struttura dei dati.
- csvfiles_manager.py:
    Gestore dei file CSV generati dal progetto (predizioni e metriche).
- newdata_manager.py:
    Gestore dei dataset passati per capire se le metriche del modello sono ancora efficienti.
    La gestione del dataset deve essere fatta in precedenza, in quanto il modello accetta solo "text" e "sentiment" come colonne di input.
- metrics.py:
    Gestore delle metriche del progetto. Qui ci sono i metodi e le librerie utili per la valutazione del modello.



## Modello utilizzato
Modello "Twitter-roBERTa-base for Sentiment Analysis" recuperato su Hugging Face, con risultati accettabili e in lingua inglese per essere utilizzato a livello globale.

## Endpoint API
- POST /predict
- GET /results
- GET /summary
- POST /model-metrics
- GET /model-metrics

## Dataset
I dataset utilizzati devono avere una struttura ben precisa:
- "text" : testo da valutare
- "sentiment" : valutazione del testo


## Metriche di valutazione del modello

Per valutare le prestazioni del modello sono state utilizzate le seguenti metriche:

### Accuracy

L’**accuracy** indica quante previsioni totali il modello ha classificato correttamente.

Esempio:
se il modello fa 100 previsioni e ne indovina 90, l’accuracy è pari al 90%.

---

### Precision

La **precision** indica quanto sono affidabili le previsioni positive del modello.

Esempio:
se il modello predice 10 recensioni come positive, ma solo 8 lo sono davvero, la precision è 80%.

---

### Recall

La **recall** indica quanti elementi positivi reali il modello riesce a trovare.

Esempio:
se ci sono 10 recensioni positive reali e il modello ne trova 7, la recall è 70%.

---

### F1-score

L’**F1-score** combina precision e recall in un unico valore.

È utile quando si vuole valutare il modello considerando sia:

* quanto sono corrette le previsioni positive;
* quanti casi positivi reali vengono trovati.

Un F1-score alto indica che il modello riesce sia a fare previsioni precise, sia a non perdere troppi casi importanti.


## Come avviare il progetto
Per avviare il progetto serve il seguente codice:

Importare tutte le librerie necessarie con requirements.txt

Da cartella del progetto:
```bash
pip install -r requirements.txt
uvicorn src/api:app --reload
```

Una volta avviato il file contenente le API, è possibile avviare anche client.py per testare un singolo sentiment. Per quanto riguarda il controllo del test su dataset di grandi dimensioni, avviare client_test.py e caricare un dataset in data/sentiment_datasets.

## Avvio con Docker

Il progetto può essere avviato anche tramite Docker, senza dover configurare manualmente l'ambiente Python locale.

### Build dell'immagine

Dalla cartella principale del progetto:

```bash
docker build -t company-sentiment .

### Avvio del container

docker run --rm -p 8000:8000 company-sentiment

L'api sarà disponibile su: http://127.0.0.1:8000

Per la documentazione automatica andare su: http://127.0.0.1:8000/docs


## Stato attuale

Progetto testato e correttamente funzionante per:
predizione del singolo sentiment, controllo delle metriche, salvataggio dello storico in file CSV sia della predizione che delle metriche.
Avviato anche progetto per il deploy. Creato Dockerfile e testato creando immagine e container.



## Sviluppi futuri
- Dashboard
- Grafana
- CD
- Deploy
- Retraining/fine-tuning
- Auto-apprendimento da sentiment ricevuti
