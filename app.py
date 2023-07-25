import os
import torch
import gradio as gr

from typing import Tuple, Dict
from timeit import default_timer as timer
from model import create_mobilenetv3


# import classes
with open('class_names.txt', 'r') as f:
    contents = f.read()
    class_names = eval(contents)
    
# import transforms and model
model, _, test_transform =create_mobilenetv3(
    num_classes=len(class_names),
    requires_grad=False)    

# load weights for model
model.load_state_dict(torch.load('model.pth', map_location='cpu')['model.state_dict'])




def predict(img) -> Tuple:
    
    # start timer
    tic = timer()

    # transform image
    transformed_img = test_transform(img).unsqueeze(0)

    # find the predicted label
    model.eval()
    with torch.inference_mode():
        y_logit = model(transformed_img)
        pred_probs = torch.softmax(y_logit, dim=1)

    # create dict with pred label and pred probs
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    
    # calculate pred time
    tac = timer()
    pred_time = round(tac-tic,4)
        
    return pred_labels_and_probs, pred_time

# create example list
examples_list = [['examples/' + image] for image in os.listdir('examples/')]
examples_list

# title - description - article
title= 'Road Sign Image Classification 🚸⛔🚦'
description = 'An Image Classification model to classify 43 different Road Signs.'
article = 'Choose between these 3 images or upload one of your own in a similar format \n(The less the background the better your accuracy will be).'

# Create Gradio App
app = gr.Interface(fn=predict,
                   inputs=gr.Image(type='pil'),
                   outputs=[
                       gr.Label(num_top_classes=3, label='predictions', ),
                       gr.Number(label='Prediction Time (seconds)')
                   ],
                   examples=examples_list,
                   title=title,
                   description=description,
                   article=article
                   )
app.launch()

