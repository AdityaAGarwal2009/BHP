from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
__model = None
__data_columns = [...]  # Your list of data columns

def load_saved_artifacts():
    global __model
    model_path = "banglore_home_prices_model.pickle"
    with open(model_path, "rb") as f:
        __model = pickle.load(f)

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

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        data = request.json
        total_sqft = float(data['total_sqft'])
        location = data['location']
        bhk = int(data['bhk'])
        bath = int(data['bath'])

        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)

        return jsonify({'estimated_price': estimated_price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_saved_artifacts()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

