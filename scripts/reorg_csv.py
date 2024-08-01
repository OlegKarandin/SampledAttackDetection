"""
This script will take the original csv file and reorganize it according to the timestamp column
Should be run in root directory of the project
"""
import pandas as pd

csv_path = "./data/Wednesday.csv"
csv_df = pd.read_csv(csv_path)
# Reorganize the entire dataframe according to 'timestamp' column
df = csv_df.sort_values(by="timestamp")
# Save the reorganized dataframe
df.to_csv("./data/Wednesday_sorted.csv", index=False)
