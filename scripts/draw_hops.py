import pandas as pd
import matplotlib.pyplot as plt

# Assuming the CSV file is in the same directory as the script
csv_file = 'results/hops.csv'

# Load the data from the CSV file
data = pd.read_csv(csv_file)

# Extract the model names and the accuracy values
models = data['model']
actor = data['actor'].str.split('±').str[0].astype(float)
squirrel = data['squirrel'].str.split('±').str[0].astype(float)
chameleon = data['chameleon'].str.split('±').str[0].astype(float)
pubmed = data['pubmed'].str.split('±').str[0].astype(float)

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(models, actor, marker='s', label='Actor')
plt.plot(models, squirrel, marker='^', label='Squirrel')
plt.plot(models, chameleon, marker='d', label='Chameleon')
plt.plot(models, pubmed, marker='o', label='Pubmed')

# Add labels
plt.xlabel('Layers', fontsize=28)
plt.ylabel('Accuracy', fontsize=28)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=20)

# Adjust layout to minimize white space
plt.tight_layout()


# Show the plot
plt.savefig('results/hops.pdf')
