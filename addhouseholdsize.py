import pandas as pd

# Load the input CSV file
input_csv_file = './old_benchmarking/testing_data_columbus_cols.csv'  # Replace with your actual input file path
data = pd.read_csv(input_csv_file)

# Add a new column called 'household_size' and initialize it to 1
data['household_size'] = 1

# Reorder the columns to have 'household_size' between 'income' and 'vehicles'
data = data[['index', 'location', 'income', 'household_size', 'vehicles', 'workers']]

# Save the updated DataFrame to a new CSV file
output_csv_file = './old_benchmarking/testing_data_columbus_complete.csv'  # Replace with your desired output file path
data.to_csv(output_csv_file, index=False)

print("New column 'household_size' added and saved to", output_csv_file)
