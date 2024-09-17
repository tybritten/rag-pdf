import json
import os

def convert_json_to_list(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def concat_dict_values_to_file(json_list, output_dir):
    output_file = os.path.join(output_dir, 'concatenated_dict.json')
    with open(output_file, 'w') as outfile:
        for contdict in json_list:
            outfile.write(contdict["content"])

def iterate_over_dir(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            data = convert_json_to_list(os.path.join(root, file))
            concat_dict_values_to_file(data, output_dir)

def main():
    input_dir = 'input/'
    output_dir = 'output/'
    iterate_over_dir(input_dir,output_dir)
