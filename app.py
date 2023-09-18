from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
import torch

app = Flask(__name__)

# Load your Deep Learning model
model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the image


# Define a function to preprocess the image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    if image_file:
        image = Image.open(image_file)
        input_image = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_image)
        
        # Process the output based on your model's architecture
        # For example, if it's a classification model, you might do:
        _, predicted_class = torch.max(output, 1)
        class_names = ['asphalt', 'bricks']  # Replace with your class names
        prediction = class_names[predicted_class]
        
        return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(debug=True)
