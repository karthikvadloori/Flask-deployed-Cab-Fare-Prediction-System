## Cab-Fare-Prediction-Flask-Deployment
This is a Cab Fare Prediction ML project deployed on production using Flask API.

### Prerequisites
You must have Scikit Learn, Pandas and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py
2. app.py 
3. request.py
4. templates

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

Enter valid numerical values in all 5 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited fare and distance.

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
