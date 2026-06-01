import requests

url = ""

data = "shit company"

response = requests.post(url=url , json=data)

if response == 200:
    print(response)
else:
    print(f"errore: {response.status_code} - {response.json}")
    