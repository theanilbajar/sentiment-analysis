from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from typing import List

from fastapi import Response


from pydantic import BaseModel
import json


app = FastAPI()

# load model
model = joblib.load('model.joblib')

# for health check
@app.get('/')
def main():
    return {'message': 'App server is working fine'}

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


class Item(BaseModel):
    """For request of the batch prediction"""
    text: str

class ItemResponse(BaseModel):
    """For response of batch prediction"""
    text:str
    sentiment: str

# endpoint to handle json file with
@app.put("/predict-batch/")
def predict_batch(items: List[Item]):

    try:
        res = []
        for item in items:

            # get prediction and add to result list
            output = model.predict([item.text])
            message = 'positive'
            if output[0] == 0:
                message = 'negative'    

            resp = ItemResponse(text=item.text, sentiment=message)
            res.append(resp.__dict__)

        # convert result to json and return the result as json
        res = json.dumps(res)
    except Exception as e:
        res = {"Failed to get predictions. Please consult with App Development Team"}
        
    return Response(content=res, media_type="application/json")
