#script to add worker column
import pandas as pd

# Load the input CSV file
input_csv_file = '../old_benchmarking/testing_data_columbus.csv'  # Replace with your actual input file path
data = pd.read_csv(input_csv_file)

# Add a new column called 'workers' and initialize it to 1
data['workers'] = 1

# Save the updated DataFrame to a new CSV file
output_csv_file = '../old_benchmarking/testing_data_columbus_cols.csv'  # Replace with your desired output file path
data.to_csv(output_csv_file, index=False)

print("New column 'workers' added and saved to", output_csv_file)
