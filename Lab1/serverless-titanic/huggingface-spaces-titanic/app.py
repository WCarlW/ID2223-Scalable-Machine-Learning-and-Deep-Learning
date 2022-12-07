import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(Pclass, Sex, Age, SibSp):
    input_list = []
    input_list.append(Pclass)
    input_list.append(Sex)
    input_list.append(Age)
    input_list.append(SibSp)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    # flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    # img = Image.open(requests.get(flower_url, stream=True).raw)            
    # return img
    if (res[0] == 0):
        result = "I'm sorry, the person is dead"
    else:
        result = "Awesome, the person is survived!!!!!!"
    return result
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Predictive Analytics",
    description="Experiment with Passenger class/Sex/Age/SibSp to predict if the person is survived or not.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Pclass (Flight class 1/2/3)"),
        gr.inputs.Number(default=1.0, label="Sex (male=1/female=2)"),
        gr.inputs.Number(default=1.0, label="Age (in years)"),
        gr.inputs.Number(default=1.0, label="SibSp (number of siblings)"),
        ],
    outputs=gr.Textbox(label="Result: "))

demo.launch()

