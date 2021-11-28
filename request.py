import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'pickup_latitude':17.4813378,'pickup_longitude':78.5240778,'dropoff_latitude':17.4444674,'dropoff_longitude':78.5038291,'passenger_count':1,'fare_amount':0})

print(r.json())