# dockerfile

# scelta di python utilizzato.

FROM python:3.11-slim

# cartella interna del container

WORKDIR /app

# copia del file utile per importazione delle librerie .->nella cartella del docker attuale

COPY requirements_docker.txt .

# scelta di scaricare alla partenza della creazione dell'immagine docker solo la libreria Torch CPU

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# lancio dell'installaizone delle librerie utili tramite pip

RUN pip install -r requirements_docker.txt

# copia di tutta la struttura del progetto in docker image -> per farlo basta inserire il doppio punto. Mantiene struttura intatta

COPY . . 

# lancio della api

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
