import pandas as pd
import os

# Paths to CSV files
history_csv_path = 'path/to/history.csv'
restaurant_new_csv_path = 'path/to/restaurant_new.csv'
restaurant_b_csv_path = 'path/to/restaurant_b.csv'

# Paths to JSON files
history_json_path = 'path/to/history.json'
restaurant_new_json_path = 'path/to/restaurant_new.json'
restaurant_b_json_path = 'path/to/restaurant_b.json'

# Load CSV files into DataFrames
history_df = pd.read_csv(history_csv_path)
restaurant_new_df = pd.read_csv(restaurant_new_csv_path)
restaurant_b_df = pd.read_csv(restaurant_b_csv_path)

# Convert DataFrames to JSON and save them
history_df.to_json(history_json_path, orient='records', indent=4)
restaurant_new_df.to_json(restaurant_new_json_path, orient='records', indent=4)
restaurant_b_df.to_json(restaurant_b_json_path, orient='records', indent=4)

print("CSV files successfully converted to JSON and saved to disk.")