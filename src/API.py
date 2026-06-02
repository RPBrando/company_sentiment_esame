from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import schema
import model
import pandas as pd

app = FastAPI() # creazione API
def read_file():
    try:
        with open("results.csv", "r") as file:
            df = pd.read_csv(file)
    except:
        with open("results.csv", "w") as file:
            df = pd.DataFrame(
                 columns=["text", "label", "score"]
            )

    return df

def write_file(df):
    with open("results.csv", "w") as file:
        df.to_csv(file, index = False)
        
    print(f"file scritto correttamente")


@app.get("/", response_class=HTMLResponse) # prova del get
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

@app.post("/predict") # prova del post
async def predicton(text: schema.Sentiment):
        testo = text.sentiment
        predict = model.prediction(testo)
        print(predict)

        df_prediction = pd.DataFrame([{
             "text": testo,
             "label": predict[0]["label"],
             "score": predict[0]["score"]
        }])

        df = read_file()

        df_total = pd.concat([df,df_prediction])
        write_file(df_total)
        return predict

@app.get("/results") #prova del get dopo il risultato
async def showing_prediction():
    df = read_file()
    return df.to_dict(orient="records")


if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, "0.0.0.0", port= 8000)


        
