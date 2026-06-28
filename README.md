# company_sentiment

## Obiettivo del progetto

Controllare andamento del sentient aziendale online. Creare pipeline di CI CD in modo 
da garantire qualità in tutte le fasi di sviluppo e deployment.
Controllare in modo costante con inplimentazione DB esterni:
- andamento delle performance del modello utilizzato
- valutazione della necessità di eventuale apprendimento

Creazione di metodi di auto apprendimento

## Funzionalità principali
- Analisi sentiment su testo singolo
- Salvataggio storico predizioni
- Lettura dataset pubblico
- Valutazione modello
- Salvataggio metriche nel tempo
- Auto apprendimento da sentiment ricevuti

## Architettura del progetto
Spiegare a cosa servono:
- API.py:
    contenitore di tutti gli endpoint presenti nel progetto.
    Qui presenti metodi per leggere/scrivere metriche e analisi sentiment fatti
- model.py:
    Contenitore del modello e del metodo di predizione
- schema.py:
    Contenitore delle classi Pydantic per il controllo della struttura dei dati
- csvfiles_manager.py:
    gestore dei file csv generati dal progetto (predizioni e metriche)
- newdata_manager.py:
    gestore dei dataset passati per capire le metriche del modello se son ancora efficienti.
    La gestione del dataset deve essere fatto in precedenza in quanto il modello accetta solo "text" e "sentiment" come colonne di input
- metrics.py:
    Gestore delle metriche del progetto. Qui ci sono i metodi e le librerie utili per la valutazione del modello



## Modello utilizzato
Modello "Twitter-roBERTa-base for Sentiment Analysis" recuperato su Hugging Face con ottimi con risultati accettabili e in lingua inglese per essere utilizzato a livello globale.

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

## Metriche
Spiegare precision, recall, f1-score, accuracy in modo semplice.

## Come avviare il progetto
Per avviare il progetto serve il seguente codice:

Da cartella del progetto:
uvicorn src/api:app --reload

una volta avviato il file contentente le api è necessario avviare anche client.py per un eventuale singolo sentiment e per controllare la valutazione. Per quanto riguarda il controllo del test su dataset di grande dimensioni avviare client_test.py e caircare un dataset in data/sentiment_datasets


## Stato attuale
Progetto testato e correttamente funzionante per:
predizione del singono sentiment, controllo delle metriche, salvataggio dello strorico in file csv sia della predizione che delle metriche

## Sviluppi futuri
- Dashboard
- Grafana
- CI/CD
- Deploy
- Retraining/fine-tuning