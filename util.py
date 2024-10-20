import json

__locations = None

def get_location_names():
    load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    global __locations
    print("loading saved artifacts...start")
    with open("columns.json", "r") as f:
        json_data = json.load(f)
        __locations = json_data['data_columns'][3:]  # Adjust this based on your JSON structure
    print("loading saved artifacts...done")
