from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
__model = None
__data_columns = ["total_sqft", "bath", "bhk"] + [
    "1st block jayanagar", "1st phase jp nagar", "2nd phase judicial layout", 
    # ... (add all other location names here)
    "yeshwanthpur"
]  # Your list of data columns, including location names

def load_saved_artifacts():
    global __model
    model_path = "banglore_home_prices_model.pickle"
    with open(model_path, "rb") as f:
        __model = pickle.load(f)
        print("Model loaded successfully.")

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        print(f"Invalid location: {location}. Valid locations are: {__data_columns[3:]}")
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[__data_columns.index("total_sqft")] = sqft
    x[__data_columns.index("bath")] = bath
    x[__data_columns.index("bhk")] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = __model.predict([x])[0]
    return round(predicted_price, 2)

def get_location_names():
    return __data_columns[3:]  # This returns all location names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_location_names', methods=['GET'])
def get_location_names_endpoint():
    try:
        locations = get_location_names()
        print(f"Locations fetched: {locations}")  # Log fetched locations
        return jsonify({'locations': locations})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        data = request.json
        total_sqft = float(data['total_sqft'])
        location = data['location']
        bhk = int(data['bhk'])
        bath = int(data['bath'])

        print(f"Received data: sqft={total_sqft}, location={location}, bhk={bhk}, bath={bath}")

        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)

        return jsonify({'estimated_price': estimated_price})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_saved_artifacts()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
