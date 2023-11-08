import os
import json
import argparse

def get_all_file_paths(root_folder):
    file_paths = []  # List to store file paths
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)  # Combine the path and file name
            file_paths.append(file_path)
    return file_paths

def save_paths_to_jsonl(file_paths, jsonl_filename):
    with open(jsonl_filename, 'w') as outfile:
        for path in file_paths:
            json_record = json.dumps({"file_path": path})  # Create a JSON string
            outfile.write(json_record + '\n')  # Write the JSON string as a new line in the file

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument("--input_folder", help="The input folder containing files.")
args = parser.parse_args()

# Use the arguments to get the file paths and save them
all_file_paths = get_all_file_paths('/f_ndata/zekai/render_imgs/'+args.input_folder)
save_paths_to_jsonl(all_file_paths,'/f_ndata/zekai/test_json/'+args.input_folder+'.jsonl')