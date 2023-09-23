from flask import Flask, render_template, request,session,redirect,url_for,jsonify,flash
import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms

app = Flask(__name__)
app.secret_key="mani is my best friend"



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
        #cl_pr,cl_index,slip,rough=model(tensor)
        slip,rough=model(tensor)
        return render_template('index.html', slip=slip, rough=rough)
    #return jsonify({'Slipperiness': slip , 'Roughness': rough,'Class': cl_index,'Class Probability': cl_pr})
    #return jsonify({'Slipperiness': slip , 'Roughness': rough})
    return flash("Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)