import requests

url = "http://127.0.0.1:8000/test"

params ={
    "sample_size": 10000
}

print(params)

response = requests.post(url, params=params)

if response.status_code == 200:
    print(response.text)
else:
    print(f"errore: {response.status_code} - {response.text}")
