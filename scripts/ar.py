import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("results/nrl-f.csv")

# Function to extract the mean value from the "mean±std" format
def extract_mean(value):
    return float(value.split('±')[0])

# Apply the function to all columns except the first one (which contains model names)
mean_values = df.iloc[:, 1:].applymap(extract_mean)

# Calculate the rank of each model for each column (ascending rank means lower values are better)
ranks = mean_values.rank(ascending=False)

# Calculate the average rank across all datasets
df['Average Rank'] = ranks.mean(axis=1)

print(ranks)

# Save the updated DataFrame to a new CSV file
df.to_csv("results/nrl-f_with_avg_rank.csv", index=False)

# # Display the updated DataFrame
# print(df)
