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

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    print(f"Predicting with: {x}")
    predicted_price = __model.predict([x])[0]
    print(f"Predicted price: {predicted_price}")
    return round(predicted_price, 2)

def get_location_names():
    return __locations

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

@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    try:
        if request.method == 'POST':
            total_sqft = float(request.form['total_sqft'])
            location = request.form['location']
            bhk = int(request.form['bhk'])
            bath = int(request.form['bath'])
        elif request.method == 'GET':
            total_sqft = float(request.args.get('total_sqft'))
            location = request.args.get('location')
            bhk = int(request.args.get('bhk'))
            bath = int(request.args.get('bath'))

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
