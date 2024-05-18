import json

def write_to_json(data, file_path):
    """
    Write data to a JSON file.

    Parameters:
    - data: The data to be written to the file.
    - file_path: The path to the JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage:
data_to_write = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}

file_path = '../filters4_xi0/clk_1536000/parameters/ecg/output.json'
write_to_json(data_to_write, file_path)