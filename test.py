import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://h7e9x4v2c3.execute-api.us-east-1.amazonaws.com/test/predict'
data = {'url': 'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41477-019-0374-3/MediaObjects/41477_2019_374_Figa_HTML.jpg'}
result = requests.post(url, json=data).json()
print(result)