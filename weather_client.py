import requests
import json

weather_url = 'https://api.open-meteo.com/v1/forecast?latitude=50.100&longitude=22.050&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m&daily=windspeed_10m_max&timezone=Europe%2FBerlin&past_days=1'
airly_api_url = 'https://airapi.airly.eu/v2/measurements/installation?installationId=7491'

headers = {'Accept': 'application/json',
           'Accept-Encoding': 'gzip',
           'apikey': '07nP6PqsOPKPYMOq2L0D8qlFpcvYyY2f'
           }
weather_res = requests.get(weather_url)
airly_res = requests.get(airly_api_url, headers=headers)

print(weather_res.status_code)  # 200 (hopefully)
print(airly_res.status_code)  # 200 (hopefully)

# print(airly_res.json()['history'])
# theMap = map(lambda x: , airly_res.json()['history'])

# print(r.text)  # { "image": "image.jpg", "link": "link.com" }
# print(r.json())
# print(json.dumps(json.loads(r.text), indent=2))


# &past_days=1



history = airly_res.json()['history']
forecast = airly_res.json()['forecast']

# weather = weather_res.json()['hourly']

data = []

for val in history:
    # data.append({"date": val['fromDateTime'], "pm25": val['tillDateTime']})
    # data.append({"date": val['fromDateTime'], "pm25": val['values'][0]['value']})
    # data.append({"date": val['fromDateTime'], "pm25": val['values']})
    print('this is my value', val['values'][0])

# for val in forecast:
#     # print(val['fromDateTime'])
#     data.append({"date": val['fromDateTime'], "pm25": val['values'][0]['value']})

# print(data)

# for val in data:
#     print(val['pm25'])

# print(weather)