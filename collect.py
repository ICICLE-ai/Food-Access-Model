import os
import json
import re

def load_json_files(directory):
    json_data = []
    pattern = re.compile(r'^step\.speedscope(_\d+)?\.speedscope\.json$')
    for filename in os.listdir(directory):
        if pattern.match(filename):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                json_data.append(data)
    return json_data

def process_json_data(json_data):
    processed_data = []
    for data in json_data:
        frames = data.get('shared', {}).get('frames', [])
        profiles = data.get('profiles', [])
        
        # Find the index of the frame with name 'run_endpoint_function'
        run_endpoint_function_index = next((i for i, frame in enumerate(frames) if frame['name'] == 'run_endpoint_function'), None)
        
        if run_endpoint_function_index is not None and run_endpoint_function_index + 1 < len(frames):
            # Get the next frame which should be 'step'
            step_frame = frames[run_endpoint_function_index + 1]
            
            if step_frame['name'] == 'step':
                step_frame_index = run_endpoint_function_index + 1
                
                # Collect events with the frame attribute matching the step_frame_index
                for profile in profiles:
                    events = profile.get('events', [])
                    for event in events:
                        if event.get('frame') == step_frame_index:
                            processed_data.append(event)
    
    return processed_data

if __name__ == "__main__":
    directory = 'profiling/api'
    json_files_data = load_json_files(directory)
    processed_data = process_json_data(json_files_data)
    print(processed_data)