from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model
import pandas as pd

app = FastAPI() # creazione API

def read_file(): # funzione per leggere file - controllo dell'esistenza del file
    try:
        with open("results.csv", "r") as file:
            df = pd.read_csv(file)

    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["text", "label", "score"]
        )

    except pd.errors.EmptyDataError:
        df = pd.DataFrame(
                columns=["text", "label", "score"]
        )

    return df

def write_file(df): # scrittura del file
    
    with open("results.csv", "w") as file:
        df.to_csv(file, index = False)
        
    print(f"file scritto correttamente")


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
        predict = model.prediction(testo)   # recupero della predizione del modello
        print(predict)                      # stampa della predizione

        df_prediction = pd.DataFrame([{     # creazione del df con la sola stringa della predizione
             "text": testo,
             "label": predict[0]["label"],
             "score": predict[0]["score"]
        }])

        df = read_file()                    # lettura del dataframe dal file

        df_total = pd.concat([df,df_prediction], ignore_index=True) # concatenazione dei 2 dataframe

        write_file(df_total)                # scrittura del file con il dataframe totale
        return predict                     

@app.get("/results") # stampa del file nel web
async def showing_prediction():
    df = read_file()
    return df.to_dict(orient="records")


# partenza del file con uvicorn

if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, "0.0.0.0", port= 8000)


        
