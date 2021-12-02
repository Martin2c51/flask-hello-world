
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request
from flask_cors import CORS
from flask import render_template
#from fastai.vision.all import *
import pathlib


try:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
except Exception as e:
    print(e)

def GetLabel(fileName):
  return fileName.split('-')[0]

try:
    learn = load_learner('export.pkl') #Import Model
except:
    pass

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index2.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = PILImage.create(request.files['file'])
    label,_,probs = learn.predict(img)
    return f'{label} ({torch.max(probs).item()*100:.0f}%)'


