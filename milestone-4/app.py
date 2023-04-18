from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('model.joblib')
# Define prediction function
def return_prediction(input_data):
    df = pd.DataFrame(input_data).T
    return model.predict(df).tolist()

@app.route("/")
def index():

    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """


@app.route('/predict', methods=['POST'])
def rainfall_prediction():
    content = request.json
    prediction = return_prediction(content)
    results = prediction

    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port="8080")
