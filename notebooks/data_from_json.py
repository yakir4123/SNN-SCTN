import os
import json


# Global Variables
JSON_FILE_PATH = "../filters4_xi0/clk_1536000/parameters/ecg/lf4"

def get_data_from_json():
# Iterate through each file in the folder
    for filename in os.listdir(JSON_FILE_PATH):
        if filename.endswith('.json'):
            file_path = os.path.join(JSON_FILE_PATH, filename)

            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)


            weight_results = data.get('weight_results')
            theta_results = data.get('theta_results')
            input_freq0 = data.get('input_freq0')
            # Print the result or perform further processing
            print(f"The weight_results in {filename} is: {weight_results}")
            print(f"The theta_results in {filename} is: {theta_results}")
            print(f"The input_freq0 in {filename} is: {input_freq0}")
            #return weight_results, theta_results

get_data_from_json()