#### Bar plots of the results

################################################################################
#### (1) Prediction per variety
################################################################################

#### R^2 distribution
################################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data: Only R² values
data = {
    'Target Variable': ['Soluble Protein', 'S/T', 'FAN', 'Alpha - Amylase ', 'pH', 'Blue'],
    'R²': [0.79, 0.71, 0.73, 0.64, 0.72, 0.75]
}

df = pd.DataFrame(data)

# Sort by R²
df = df.sort_values(by='R²', ascending=True)

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Barplot
colors = sns.color_palette("Set2")
ax = sns.barplot(x='Target Variable', y='R²', data=df, palette=colors)

# Add value labels above bars
for i, r2 in enumerate(df['R²']):
    ax.text(i, r2 + 0.02, f"{r2:.2f}", ha='center', va='bottom', fontsize=10)

# Labels and title
plt.ylabel('R² Score')
plt.title('R² Score per Target Variable (Regression)')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()



#### Seed part, Filter type distribution in Dataset selected
################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Input: List of full dataset strings
# ---------------------------------------
datasets = [
    'Endosperm part, Erosion 1, Filter SG_SD',
    'Full seed, Erosion 1, Filter SG_SD',
    'Endosperm part, Erosion 1, Filter MSC_SNV',
    'Full seed, Erosion 5, Filter SG_SD',
    'Endosperm part, Erosion 1,  Filter SG_SD',
    'Endosperm part, Erosion 1,  Filter SG_SD'
]

# Convert to DataFrame
df = pd.DataFrame(datasets, columns=['Dataset'])

# ---------------------------------------
# Extract seed part and filter type
# ---------------------------------------
df['Seed Part'] = df['Dataset'].str.extract(r'^(Full seed|Endosperm part)')
df['Filter Type'] = df['Dataset'].str.extract(r'Filter\s+(\w+_\w+|\w+)')

# ---------------------------------------
# Plot 1: Count of seed parts
# ---------------------------------------
plt.figure(figsize=(6, 4))
seed_part_counts = df['Seed Part'].value_counts()
sns.barplot(x=seed_part_counts.index, y=seed_part_counts.values, palette="Set2")
plt.title('Distribution of Selected Datasets by Seed Part')
plt.ylabel('Count')
plt.xlabel('Seed Part')
plt.tight_layout()
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.show()

# ---------------------------------------
# Plot 2: Count of filter types
# ---------------------------------------
plt.figure(figsize=(6, 4))
filter_counts = df['Filter Type'].value_counts()
sns.barplot(x=filter_counts.index, y=filter_counts.values, palette="Set2")
plt.title('Distribution of Selected Datasets by Spectral Filter')
plt.xlabel('Spectral Filter')
plt.ylabel('Count')
plt.xlabel('Filter Type')
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()



#### Feature selection distribution
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Input: List of selected feature selection methods
# ---------------------------------------
feature_methods = [
    'No selection',
    'No selection',
    'SPA',
    'PCA',
    'SPA',
    'No selection'
]

# Convert to DataFrame for counting
df = pd.DataFrame(feature_methods, columns=['Feature Selection Method'])

# Count how many times each method appears
method_counts = df['Feature Selection Method'].value_counts()

# ---------------------------------------
# Plotting the bar chart
# ---------------------------------------
sns.set(style="whitegrid")  # Clean background style
plt.figure(figsize=(8, 5))  # Set figure size
colors = sns.color_palette("Set2")  # Choose a color palette

# Create the bar plot
ax = sns.barplot(x=method_counts.index, y=method_counts.values, palette=colors)

# Add count labels above bars
for i, count in enumerate(method_counts.values):
    plt.text(i, count + 0.1, str(count), ha='center')

# Add titles and axis labels
plt.title('Distribution of Selected Feature Selection Methods')
plt.xlabel('Feature Selection Method')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.tight_layout()  # Make sure everything fits

# Show the plot
plt.show()



#### Models distribution
################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Input: List of selected models
# ---------------------------------------
models = [
    'SVR',
    'SVR',
    'SVR',
    'SVR',
    'PLSR',
    'SVR'
]

# Convert to DataFrame for counting
df = pd.DataFrame(models, columns=['Model'])

# Count how many times each model appears
model_counts = df['Model'].value_counts()

# ---------------------------------------
# Plotting the bar chart
# ---------------------------------------
sns.set(style="whitegrid")  # Clean background style
plt.figure(figsize=(8, 5))  # Set figure size
colors = sns.color_palette("Set2")  # Choose a color palette

# Create the bar plot
ax = sns.barplot(x=model_counts.index, y=model_counts.values, palette=colors)

# Add count labels above bars
for i, count in enumerate(model_counts.values):
    plt.text(i, count + 0.1, str(count), ha='center')

# Add titles and axis labels
plt.title('Distribution of Selected Models')
plt.xlabel('Model')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.tight_layout()  # Make sure everything fits

# Show the plot
plt.show()



################################################################################
#### (2) Prediction per seed
################################################################################

#### Bar plot of R2 on CNN
########################################################


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'Target Variable': ['Blue', 'Soluble Protein', 'S/T', 'FAN', 'Diast. Power'],
    'R²': [0.70, 0.75, 0.71, 0.75, 0.77],
    'R² (Mean per Variety)': [0.73, 0.79, 0.72, 0.78, 0.80]
}

df = pd.DataFrame(data)

# Sort by R² (Mean per Variety)
df = df.sort_values(by='R² (Mean per Variety)', ascending=True)

# Set style
sns.set(style="whitegrid")

# Set figure size
plt.figure(figsize=(10, 6))

# Plot the grouped bar chart
bar_width = 0.4
x = range(len(df))
colors = sns.color_palette("Set2")

bars1 = plt.bar(x, df['R²'], width=bar_width, label='R²', color=colors[0])
bars2 = plt.bar([p + bar_width for p in x], df['R² (Mean per Variety)'], width=bar_width, label='R² (Mean per Variety)', color=colors[1])

# Add value labels above bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center')

for i, bar in enumerate(bars2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center')

# Add x-axis labels
plt.xticks([p + bar_width / 2 for p in x], df['Target Variable'], rotation=45)
plt.ylabel('R² Score')
plt.title('Comparison of R² vs. R² (Mean per Variety)')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

















