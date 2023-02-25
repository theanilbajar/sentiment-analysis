from fastapi import FastAPI
import joblib
import pickle
from pydantic import BaseModel
from typing import Union

# Declaring our FastAPI instance
app = FastAPI()

model = joblib.load('model.joblib')

# for health check
@app.get('/')
def main():
    return {'message': 'I am working'}

# endpoint for single review sentiment
@app.post("/predict/")
async def predict(text: str):

    try:
        output = model.predict([text])

        message = 'positive'

        if output[0] == 0:
            message = 'negative'

    except Exception as e:
        message = 'Some error occurred. Please contact the API Development team'

    return message