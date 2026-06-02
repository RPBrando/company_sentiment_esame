import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "sentiment" : "normal company"
}

response = requests.post(url=url , json=data)

if response.status_code == 200:
    print(response.text)
else:
    print(f"errore: {response.status_code} - {response.json}")
