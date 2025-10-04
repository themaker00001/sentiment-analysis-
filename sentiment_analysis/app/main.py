from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
sentiment_model=pipeline("sentiment-analysis")

app=FastAPI()

class Review(BaseModel):
    text:str
    
class Feedback(BaseModel):
    text:str
    user_label:str

feedback_store=[]
@app.get("/")
def getting_home():
    return {"message":"at the home page"}

@app.post("/predict")
def prediction(review_text:Review):
    response=sentiment_model(review_text.text)[0]
    return {"label":response["label"],"score":response["score"]}

@app.post("/feedback")
def feedback(feedback:Feedback):
    feedback_store.append(feedback)
    return {"message":"feedback recieved ","total_feedback":len(feedback_store)}

@app.get("/get_feedback")
def get_all_feedback():
    return {"feedback_store":feedback_store}
