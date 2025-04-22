import os
import json
import re
import pandas as pd

def load_json_files(directory):
    json_data = []
    pattern = re.compile(r'^step\.speedscope(_\d+)?\.speedscope\.json$')
    
    for filename in os.listdir(directory):
        if pattern.match(filename):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                
                # Add filename to each JSON object
                if isinstance(data, dict):  # If the JSON is a dictionary
                    data["name_file"] = filename
                elif isinstance(data, list):  # If the JSON is a list of dictionaries
                    for item in data:
                        if isinstance(item, dict):
                            item["name_file"] = filename
                
                json_data.append(data)

    return json_data

def process_json_data(json_data):
    processed_data = []
    
    for data in json_data:
        if isinstance(data, dict) and "profiles" in data:
            for profile in data["profiles"]:
                if isinstance(profile, dict) and "endValue" in profile:
                    processed_data.append({
                        "name_file": data.get("name_file", "unknown"),
                        "endValue": profile["endValue"]
                    })
    
    return processed_data

def save_to_excel(data, output_file="profiling_results.xlsx"):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    
if __name__ == "__main__":
    directory = 'profiling/api'
    json_files_data = load_json_files(directory)
    processed_data = process_json_data(json_files_data)
    #print(processed_data)
    save_to_excel(processed_data)