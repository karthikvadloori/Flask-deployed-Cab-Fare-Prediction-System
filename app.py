import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        if to_radians:
            lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        a = np.sin((lat2-lat1)/2.0)**2 +  np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
        return earth_radius * 2 * np.arcsin(np.sqrt(a))
    b=haversine(int_features[0],int_features[1],int_features[2],int_features[3])
    final_features = pd.DataFrame({'pickup_latitude':int_features[0],'pickup_longitude':int_features[1],'dropoff_latitude':int_features[2],'dropoff_longitude':int_features[3],'passenger_count':int_features[4],'dist':b,'fare_amount':0},index=[0])
    prediction = model.predict(final_features.iloc[:,0:6])
    
    
    
    output = float(prediction)
    return render_template('index.html', prediction_text1='The Estimated Cab Fare: $ {}'.format(output),prediction_text2='The Distance: {} KMs'.format(np.round((b*1.255),2)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    
    prediction = model.predict(pd.DataFrame(data,index[0]).iloc[:,0:6])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)