import os
from Questgen import main
from pprint import pprint

import requests

def papago_translate(text, source='ko', target='en', client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET'):
    url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    data = {
        "source": source,
        "target": target,
        "text": text
    }
    response = requests.post(url, headers=headers, data=data)
    result = response.json()
    return result['message']['result']['translatedText']

payload = {
    "input_text": "The capital of France is Paris. Paris is known for its art, fashion, and culture. The Eiffel Tower is one of the most famous landmarks in Paris.",
}

qg = main.QGen()
output = qg.predict_mcq(payload)
output = [output[0]['question_statement'], output[0]['answer']]
output = []
pprint(output)