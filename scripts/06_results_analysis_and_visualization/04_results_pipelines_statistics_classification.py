import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
#### (3) Classification per variety
################################################################################

#### ACCURACY distribution
################################################################################

data = {
    'Target Variable': [
        'Protein', 'Moisture_1', '4mL', '8mL', 'Blue', 'Moisture_2', 'Friab.', 'Fine Extract', 'Colour',
        'ßeta- Glucan', 'Soluble Protein', 'Total Protein', 'S/T', 'FAN', 'Diast. Power',
        'Alpha- Amylase', 'pH', 'PUG', 'WUG'
    ],
    'ACC': [
        0.77, 0.69, 0.71, 0.83, 0.80, 0.77, 0.77, 0.74, 0.89, 0.87,
        0.89, 0.68, 0.86, 0.90, 0.83, 0.78, 0.86, 0.80, 0.76
    ]
}

df = pd.DataFrame(data)

# Sort by ACC
df = df.sort_values(by='ACC', ascending=True)

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
colors = sns.color_palette("Set2")
ax = sns.barplot(x='Target Variable', y='ACC', data=df, palette=colors)

# Add labels at the top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2)

plt.ylabel('Accuracy')
plt.title('Classification Accuracy per Target Variable')
plt.ylim(0, 1.05)  # Slightly above 1 to allow space for labels
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()








#### Seed part, Filter type distribution in Dataset selected
################################################################################

#### Seed part
################################################################################


datasets = [
    'Embryo part, Erosion 1, No Filter',
    'Embryo part, Erosion 1, Filter MSC_SNV',
    'Embryo part, Erosion 1, Filter SG_SNV',
    'Embryo part, Erosion 1, Filter SG_SNV',    
    'Full seed, Erosion 1, Filter SG_MSC',
    'Endosperm part, Erosion 1, Filter SG_FD_SNV',
    'Endosperm part, Erosion 1, Filter SG_FD',
    'Full seed, Erosion 1, Filter SG_SNV',
    'Full seed, Erosion 7, Filter SG_FD',
    'Full seed, Erosion 5, Filter SG_SD',
    'Embryo part, Erosion 1, Filter SG_FD_SNV',
    'Full seed, Erosion 1, Filter MSC_SNV',
    'Index features, Erosion 1, No Filter',
    'Index features, Erosion 1, No Filter',
    'Full seed, Erosion 7, Filter SG_MSC',
    'Full seed, Erosion 3, Filter MSC_SNV',
    'Full seed, Erosion 5, Filter MSC_SNV',
    'Full seed, Erosion 3, Filter SG_FD_SNV',
    'Full seed, Erosion 3, Filter SG_FD_SNV'
]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

df = pd.DataFrame(datasets, columns=['Dataset'])
df['Seed Part'] = df['Dataset'].str.extract(r'^(Full seed|Embryo part|Endosperm part|Index features)')

plt.figure(figsize=(6, 4))
seed_counts = df['Seed Part'].value_counts()
sns.barplot(x=seed_counts.index, y=seed_counts.values, palette="Set2")
plt.title('Distribution of Selected Datasets by Seed Part')
plt.ylabel('Count')
plt.xlabel('Seed Part')
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()


#### Filter type
################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Same dataset list
datasets = [
    'Embryo part, Erosion 1, No Filter',
    'Embryo part, Erosion 1, Filter MSC_SNV',
    'Embryo part, Erosion 1, Filter SG_SNV',
    'Embryo part, Erosion 1, Filter SG_SNV',
    'Index features, Erosion 1, No Filter',
    'Full seed, Erosion 1, Filter SG_MSC',
    'Endosperm part, Erosion 1, Filter SG_FD_SNV',
    'Endosperm part, Erosion 1, Filter SG_FD',
    'Full seed, Erosion 1, Filter SG_SNV',
    'Full seed, Erosion 7, Filter SG_FD',
    'Full seed, Erosion 5, Filter SG_SD',
    'Embryo part, Erosion 1, Filter SG_FD_SNV',
    'Full seed, Erosion 1, Filter MSC_SNV',
    'Index features, Erosion 1, No Filter',
    'Full seed, Erosion 7, Filter SG_MSC',
    'Full seed, Erosion 3, Filter MSC_SNV',
    'Full seed, Erosion 5, Filter MSC_SNV',
    'Full seed, Erosion 3, Filter SG_FD_SNV',
    'Full seed, Erosion 3, Filter SG_FD_SNV'
]

# Create DataFrame
df = pd.DataFrame(datasets, columns=['Dataset'])

# Extract filter type from dataset string
df['Filter Type'] = df['Dataset'].str.extract(r'Filter\s+(\w+(?:_\w+)*)')
df['Filter Type'] = df['Filter Type'].fillna('No Filter')

# Count and plot
plt.figure(figsize=(7, 4))
filter_counts = df['Filter Type'].value_counts()
sns.barplot(x=filter_counts.index, y=filter_counts.values, palette="Set2")
plt.title('Distribution of Selected Datasets by Spectral Filter')
plt.ylabel('Count')
plt.xlabel('Filter Type')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()


#### Feature selection distribution
################################################################################


methods = [
    'No selection', 'No selection', 'PCA', 'No selection', 'No selection',
    'No selection', 'Select K Best_f', 'Select K Best_f', 'No selection',
    'No selection', 'No selection', 'No selection', 'Select K Best_f',
    'No selection', 'Select K Best_f', 'No selection', 'Select K Best_f',
    'PCA', 'No selection'
]

df = pd.DataFrame(methods, columns=['Method'])
counts = df['Method'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=counts.index, y=counts.values, palette="Set2")
for i, val in enumerate(counts.values):
    plt.text(i, val + 0.1, str(val), ha='center')
plt.title('Distribution of Selected Feature Selection Methods')
plt.ylabel('Count')
plt.xlabel('Method')
plt.tight_layout()
plt.show()


#### Models distribution
################################################################################

models = [
    'SVC', 'SVC', 'Naïve Bayes', 'SVC', 'SVC',
    'SVC', 'Naïve Bayes', 'XGB', 'SVC', 'SVC',
    'XGB', 'Random Forest', 'LDA', 'Naïve Bayes', 'LDA',
    'SVC', 'LDA', 'XGB', 'XGB'
]

df = pd.DataFrame(models, columns=['Model'])
counts = df['Model'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=counts.index, y=counts.values, palette="Set2")
for i, val in enumerate(counts.values):
    plt.text(i, val + 0.1, str(val), ha='center')
plt.title('Distribution of Selected Models')
plt.ylabel('Count')
plt.xlabel('Model')
plt.tight_layout()
plt.show()



################################################################################
#### (4) Classification per seed
################################################################################

#### Bar plot of ACCURACY ON ML and CNN
########################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors (Set2 palette)
colors = sns.color_palette("Set2")

# Data from the table
data = {
    'Target Variable': ['Protein', 'Moisture_1', '4mL', '8mL', 'Blue', 'Moisture_2', 'Friab.', 'Fine Extract', 'Colour',
                        'ßeta- Glucan ppm', 'Soluble Protein', 'Total Protein', 'S/T', 'FAN', 'Diast. Power',
                        'Alpha- Amylase', 'pH', 'PUG', 'WUG'],
    'ACC': [0.79, 0.79, 0.76, 0.78, 0.79, 0.84, 0.83, 0.75, 0.88,
            0.77, 0.93, 0.70, 0.87, 0.85, 0.90, 0.79, 0.81, 0.83, 0.88],
    'ACC (Majority per Variety)': [0.79, 0.83, 0.72, 0.79, 0.89, 0.94, 0.83, 0.70, 0.94,
                                   0.83, 0.97, 0.79, 0.89, 0.89, 0.91, 0.78, 0.83, 0.86, 0.89]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by 'ACC (Majority per Variety)' for better visual ordering
df = df.sort_values(by='ACC (Majority per Variety)', ascending=True)

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 7))

# Bar width and x locations
bar_width = 0.4
x = range(len(df))

# Plot bars
plt.bar(x, df['ACC'], width=bar_width, label='ACC (Seed Level)', color=colors[0])
plt.bar([p + bar_width for p in x], df['ACC (Majority per Variety)'], width=bar_width, label='ACC (Majority Vote)', color=colors[1])

# Add value labels on top of bars
for i, (acc, acc_major) in enumerate(zip(df['ACC'], df['ACC (Majority per Variety)'])):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center', va='bottom', fontsize=9)
    plt.text(i + bar_width, acc_major + 0.01, f"{acc_major:.2f}", ha='center', va='bottom', fontsize=9)

# Set x-axis labels and formatting
plt.xticks([p + bar_width / 2 for p in x], df['Target Variable'], rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Classification Accuracy by Target Variable (Seed vs. Majority Vote)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()


#### Model Distribution Visualization (Manual Input)
########################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame manually with a 'Model' column
df = pd.DataFrame({
    'Model': [
        'CNN', 'CNN', 'CNN', 'SVC', 'SVC', 'CNN', 'CNN', 'CNN', 'SVC', 'SVC',
        'XGB', 'CNN', 'CNN', 'SVC', 'LDA', 'CNN', 'LDA', 'RF', 'CNN'
    ]
})

# Set the plot style
sns.set(style="whitegrid")

# Count the occurrences of each model
model_counts = df['Model'].value_counts()

# Create a bar plot for model distribution
plt.figure(figsize=(8, 5))
colors = sns.color_palette("Set2")
ax = sns.barplot(x=model_counts.index, y=model_counts.values, palette=colors)

# Set plot title and axis labels
plt.title("Model Distribution")
plt.ylabel("Count")
plt.xlabel("Model")
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()




