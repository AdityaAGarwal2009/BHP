from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Embed columns data directly
__data_columns = [
    "total_sqft", "bath", "bhk", "1st block jayanagar", "1st phase jp nagar",
    # ... add all your locations here ...
]
__locations = __data_columns[3:]
__model = None

def load_saved_artifacts():
    global __model
    print("loading saved artifacts...start")
    try:
        model_path = "banglore_home_prices_model.pickle"
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        with open(model_path, "rb") as f:
            __model = pickle.load(f)
        print(f"Model loaded successfully: {__model}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise e
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    print("loading saved artifacts...done")

def get_estimated_price(location, sqft, bhk, bath):
    if __model is None:
        print("Model is not loaded. Cannot predict price.")
        return None

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[__data_columns.index("total_sqft")] = sqft
    x[__data_columns.index("bath")] = bath
    x[__data_columns.index("bhk")] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    print(f"Input features for prediction: {x}")
    try:
        predicted_price = __model.predict([x])[0]
        return round(predicted_price, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_location_names', methods=['GET'])
def get_location_names_endpoint():
    try:
        response = jsonify({
            'locations': get_location_names()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        data = request.json  # Expecting application/json
        total_sqft = float(data['total_sqft'])
        location = data['location']
        bhk = int(data['bhk'])
        bath = int(data['bath'])

        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)
        response = jsonify({
            'estimated_price': estimated_price
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    load_saved_artifacts()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
