from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load your machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.json
    # Preprocess the image if necessary
    prediction = model.predict(image)  # Replace with your actual prediction code
    return jsonify({'result': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)