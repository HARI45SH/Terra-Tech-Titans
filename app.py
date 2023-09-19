from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
import torch
import os
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class Mobilenet_reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet_model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.mobilenet_model.classifier[3]=nn.Identity()
        self.classifier=nn.Linear(1024,31)
        self.regression=nn.Sequential(
            nn.Linear(1024,256),
            nn.GELU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        extraction=self.mobilenet_model(x)
        class_pred=self.classifier(extraction)
        regression_pred=self.regression(extraction)

        return class_pred,regression_pred


class Load_model():
    def __init__(self,Path):
        self.model=Mobilenet_reg()
        self.PATH = Path
        if self.PATH!=None:
            self.model.load_state_dict(self.PATH)
    
    def __call__(self,image):
        class_pred,slip_rough=self.model(image)
        slip,rough=slip_rough[0][0].item(),slip_rough[0][1].item()

        return class_pred.argmax(dim=1).item(),slip,rough


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    image_path = "uploads/" + file.filename
    file.save(filename)
    

    




    return jsonify({'Slipperiness': regression_pred , 'Roughness': slip_rough})











if __name__ == '__main__':
    m=Load_model(Path=None)
    print(m(
        torch.randn(1,3,224,224)
    ))
    app.run(port=3000, debug=True)