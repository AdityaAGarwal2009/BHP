from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and data columns
__model = None
__data_columns = []

def load_saved_artifacts():
    global __model, __data_columns
    model_path = "banglore_home_prices_model.pickle"
    with open(model_path, "rb") as f:
        __model = pickle.load(f)
        print("Model loaded successfully.")

    # Load data columns
    # Assuming you have a separate file or method to get this
    __data_columns = ["total_sqft", "bath", "bhk"] + ...  # Add other feature names here

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[__data_columns.index("total_sqft")] = sqft
    x[__data_columns.index("bath")] = bath
    x[__data_columns.index("bhk")] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = __model.predict([x])[0]
    return round(predicted_price, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    try:
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        # Make prediction
        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)
        
        return render_template('index.html', prediction_text='Estimated Price: ₹{}'.format(estimated_price))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

if __name__ == "__main__":
    load_saved_artifacts()
    app.run(debug=True)

