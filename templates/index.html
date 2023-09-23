from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import os
import numpy as np
from io import BytesIO
from torchvision import transforms

app = Flask(__name__)



# Set the upload folder
UPLOAD_FOLDER = 'uploads'
if os.path.isdir(UPLOAD_FOLDER) == False:
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from python_files.model_eval import Load_model
model=Load_model(Path=None)

app.config['model']=model

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        tensor=torch.from_numpy(np.array(Image.open(filename).resize((224,224)))).permute(2,0,1).unsqueeze(0)
        tensor=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])(tensor.float())
        cl,slip,rough=model(tensor)
        print(cl,slip,rough)
        return jsonify({'predictions': cl, 'Roughness': str(round(rough,2)), 'Slipperiness': str(round(slip,2))})


if __name__ == '__main__':
    app.run(debug=True)