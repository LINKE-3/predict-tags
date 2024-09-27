from typing import Union
#from python_dev.preparation_data import *
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re
import contractions
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()

emoticons_regex = r"(^|\s)(:\)|[\d\s]+|I|:\(|;\)|:D|;D|:p|:P|:o|:O|:@|:s|:S|:\$|:\||:\\|:\/|:\'\\(|:\'\\)|:\*|<3)($|\s)"


multilabel_binarizer = joblib.load('share/mltbbin.joblib')
tfidf = joblib.load('share/tfidf.joblib')
ovr_dt = joblib.load('share/dt.joblib')

def preprocessing(comment):
    comment = comment.lower()
    comment = contractions.fix(comment)
    comment = re.sub(r'https?://\S+', ' ', comment)
    comment = re.sub(r'[^\w$@#_+<>]', ' ', comment)
    comment = re.sub(r'\s+', ' ', comment)
    return comment

def lemmatize(comment):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in comment.split()]
    lemmatized_comment = ' '.join(lemmatized_words)

    return lemmatized_comment

def replace_characters(text):
    text = re.sub(emoticons_regex, " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text
    
def vectorize(sentence, tfidf):
    sentvect = tfidf.transform([sentence])
    return sentvect
    
def predict_sentence(sentvect, loaded_model):
    Y_pred = loaded_model.predict(sentvect)
    tag = multilabel_binarizer.inverse_transform(Y_pred)
    return tag

@app.get("/")
def read_root():
    return {"Hello": "World"}

class Textrequest(BaseModel):
    content: str

@app.post("/gettags/")

def read_item(question: Textrequest):
    textclean = preprocessing(question.content)
    question = lemmatize(textclean)
    question_clean = replace_characters(question)
    questionvec = vectorize(question_clean, tfidf)
    result = predict_sentence(questionvec, ovr_dt)
    return {"item_id": result}