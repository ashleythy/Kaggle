# import uvicorn
# from fastapi import FastAPI
# import joblib
from TitanicPassenger import TitanicPassenger

# app = FastAPI()
# joblib_in = open("car-recommender.joblib","rb")
# model=joblib.load(joblib_in)


# @app.get('/')
# def index():
#     return {'message': 'Cars Recommender ML API'}

# @app.post('/car/predict')
# def predict_car_type(data:CarUser):
#     data = data.dict()
#     age=data['age']
#     gender=data['gender']

#     prediction = model.predict([[age, gender]])
    
#     return {
#         'prediction': prediction[0]
#     }

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

from TitanicPassenger import TitanicPassenger

# test 
def preprocess(data: TitanicPassenger) -> list: 
    """
    Preprocess raw input data into format expected by model.
    """
    raw_data = data.dict() 

    return raw_data

passenger_data = TitanicPassenger(
    Age=29.5,
    SibSp=1,
    Parch=0,
    Fare=72.5,
    len_unq_firstname=6,
    len_char_firstname='10'
)

res = preprocess(data = passenger_data)

print(res)