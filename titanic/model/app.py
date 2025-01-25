import os
import pickle
import uvicorn
from fastapi import FastAPI

from ml.data import preprocess_data, inference
from ml.TitanicPassenger import TitanicPassenger

# model path 
model_dir = os.path.join(
    os.path.dirname(__file__),
    'artefact'
)

model_path = os.path.join(
    model_dir, 
    'titanic_predictor.pkl'
)

# log path 
log_dir = os.path.join(
    os.path.dirname(__file__),
    'log'
)

# configure application
app = FastAPI()

# load model 
with open(model_path, 'rb') as file:
    trained_model = pickle.load(file)

@app.get('/')
def index():
    return {'message': 'API to predict if a passenger from the Titanic shipwreck survived or not'}

@app.post('/titanic/predict')
def predict(passenger_data: TitanicPassenger):
    print(f"Validated passenger: {passenger_data}")

    # preproces passenger data
    processed_passenger_data = preprocess_data(passenger=passenger_data)

    preds = inference(model=trained_model, X_data=processed_passenger_data)

    return preds

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)