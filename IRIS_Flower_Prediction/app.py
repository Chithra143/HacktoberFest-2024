from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('iris_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define feature names
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input data into a DataFrame with feature names
    input_data = pd.DataFrame([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]], columns=feature_names)

    # Scale the features
    features_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    species_name = ['setosa', 'versicolor', 'virginica']
    predicted_species = species_name[predicted_class]

    return jsonify({'prediction': predicted_species})

if __name__ == '__main__':
    app.run(debug=True)
