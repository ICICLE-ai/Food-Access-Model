import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_endpoint_name(filename):
    # Extract the endpoint name from the filename (everything before the first speedscope)
    match = re.match(r'^(.+?)\.speedscope', filename)
    if match:
        return match.group(1)
    return None

def process_profiling_files(directory):
    # Dictionary to store times for each endpoint
    endpoint_times = defaultdict(list)
    
    # Process each speedscope file
    for filename in os.listdir(directory):
        if filename.endswith('.speedscope.json'):
            filepath = os.path.join(directory, filename)
            endpoint = extract_endpoint_name(filename)
            
            if endpoint:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'profiles' in data and len(data['profiles']) > 0:
                        # Get the endValue which represents the total execution time
                        execution_time = data['profiles'][0]['endValue']
                        endpoint_times[endpoint].append(execution_time)
    
    return endpoint_times

def create_summary_table(endpoint_times):
    # Calculate statistics for each endpoint
    summary_data = []
    for endpoint, times in endpoint_times.items():
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        num_calls = len(times)
        
        summary_data.append({
            'Endpoint': endpoint,
            'Average Time (s)': round(avg_time, 2),
            'Min Time (s)': round(min_time, 2),
            'Max Time (s)': round(max_time, 2),
            'Number of Calls': num_calls
        })
    
    # Create DataFrame and sort by average time
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Average Time (s)', ascending=False)
    return df

def create_bar_plot(df):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['Endpoint'], df['Average Time (s)'])
    
    # Customize the plot
    plt.title('API Endpoint Execution Times', pad=20)
    plt.xlabel('Endpoint')
    plt.ylabel('Average Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('endpoint_times.png')
    plt.close()

def main():
    # Directory containing the profiling files
    directory = 'profiling/api'
    
    # Process the files
    endpoint_times = process_profiling_files(directory)
    
    # Create summary table
    df = create_summary_table(endpoint_times)
    
    # Save to Excel
    df.to_excel('endpoint_times.xlsx', index=False)
    print("Summary table saved to endpoint_times.xlsx")
    
    # Create and save bar plot
    create_bar_plot(df)
    print("Bar plot saved to endpoint_times.png")

if __name__ == "__main__":
    main() 