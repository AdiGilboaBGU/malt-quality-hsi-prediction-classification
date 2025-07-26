import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#### Descriptive statistics of Y target variables
########################################################

########################################################
#### Y per speice prediction
########################################################


#--------------------------------------------------------
# Summary statistics table
#--------------------------------------------------------

# Load the Excel file (make sure it's in the same folder or provide the full path)
file_path = r'G:\My Drive\Thesis\Temp_Work\excel_files\Y_DL.xlsx'
df = pd.read_excel(file_path)

# Drop the first column (image name)
df = df.drop(columns=["Image_Name"])


# Compute descriptive statistics (mean, std, min, max, skewness, kurtosis) for each reflectance feature
summary_stats = pd.DataFrame({
    'Unique Values': df.nunique(),
    'Mean': df.mean(),
    'Median': df.median(),
    'Std': df.std(),
    'Min': df.min(),
    'Max': df.max(),
    'Range': df.max() - df.min(),
    'Skewness': df.skew(),
    'Kurtosis': df.kurt()
})

# Display the table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Apply formatting: integers stay as is, floats show up to 2 decimals only if needed
def custom_format(x):
    return f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x

formatted_summary = summary_stats.applymap(custom_format)
print(formatted_summary)



#--------------------------------------------------------
# BOX-PLOT and Histograms of each Y 
#--------------------------------------------------------

# Get list of target variables (excluding the first column with image names)
target_columns = df.columns[1:]

# Set general style for plots
sns.set(style="whitegrid")

# Loop through each target variable
for col in target_columns:
    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[col], color='#4C72B0')
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    
    # Histogram + KDE
    plt.subplot(1, 2, 2)
    sns.histplot(df[col], kde=True, color='#4C72B0')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()
    
 #--------------------------------------------------------   
# Correlation matrixs between Y
#--------------------------------------------------------

    
# Calculate correlation matrix
corr_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={'shrink': 0.8})
plt.title("Correlation Matrix of Target Variables")
plt.tight_layout()
plt.show()

#--------------------------------------------------------
# Partial Correlation matrixs between Y
#--------------------------------------------------------


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import partial_corr

# Load the Excel file (make sure it's in the same folder or provide the full path)
file_path = r'G:\My Drive\Thesis\Temp_Work\excel_files\Y_DL.xlsx'
df = pd.read_excel(file_path)

# Drop the first column (image name)
df = df.drop(columns=["Image_Name"])

# Ensure only numeric columns are used
df_numeric = df.select_dtypes(include=[np.number])

# Create an empty DataFrame to store partial correlations
variables = df_numeric.columns
partial_corr_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)

# Loop through each pair of variables
for i in variables:
    for j in variables:
        if i == j:
            partial_corr_matrix.loc[i, j] = 1.0
        else:
            # Control for all other variables except i and j
            covariates = [col for col in variables if col not in [i, j]]
            result = partial_corr(data=df_numeric, x=i, y=j, covar=covariates, method='pearson')
            partial_corr_matrix.loc[i, j] = result['r'].values[0]

# Plot heatmap of partial correlations
plt.figure(figsize=(12, 10))
sns.heatmap(partial_corr_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={'shrink': 0.8})
plt.title("Partial Correlation Matrix of Target Variables (Pearson)")
plt.tight_layout()
plt.show()


########################################################
#### Y per speice prediction
########################################################

#--------------------------------------------------------
# Class division table
#--------------------------------------------------------

import pandas as pd

# Load the original and binary datasets
original_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y.xlsx'
binary_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_classification\binary_data.xlsx'

original_df = pd.read_excel(original_path).drop(columns=["Image_Name"])
binary_df = pd.read_excel(binary_path).drop(columns=["Image_Name"])

# Create summary table
summary_data = []

for col in original_df.columns:
    if col not in binary_df.columns:
        continue

    # Combine the two versions
    merged = pd.DataFrame({'value': original_df[col], 'class': binary_df[col]})

    # Class 0
    class_0 = merged[merged['class'] == 0]['value']
    count_0 = len(class_0)
    percent_0 = round(count_0 / len(merged) * 100, 1)
    range_0 = f"{round(class_0.min(), 2)} - {round(class_0.max(), 2)}"

    # Class 1
    class_1 = merged[merged['class'] == 1]['value']
    count_1 = len(class_1)
    percent_1 = round(count_1 / len(merged) * 100, 1)
    range_1 = f"{round(class_1.min(), 2)} - {round(class_1.max(), 2)}"

    # Append row
    summary_data.append([
        col, count_0, count_1, f"{percent_0}%", f"{percent_1}%", range_0, range_1
    ])

# Create DataFrame
summary_df = pd.DataFrame(summary_data, columns=[
    "Target Variable",
    "Class 0 Count", "Class 1 Count",
    "Class 0 %", "Class 1 %",
    "Class 0 Range", "Class 1 Range"
])

# Display full table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
print(summary_df)




#--------------------------------------------------------
# BOXPLOT of 2 classes in Binary Y with the Original Y
#--------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load original (continuous) target variables
Original_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y.xlsx'
original_df = pd.read_excel(Original_path)
original_df = original_df.drop(columns=["Image_Name"])

# Load binary target variables
Binary_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_classification\binary_data.xlsx'
binary_df = pd.read_excel(Binary_path)
binary_df = binary_df.drop(columns=["Image_Name"])

# Set style
sns.set(style="whitegrid")

# Plot a boxplot for each target variable: original value vs binary class
for column in original_df.columns:
    data = pd.DataFrame({
        'Original': original_df[column],
        'Class': binary_df[column].astype(str)  # Convert class to string for category color mapping
    })

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Class', y='Original', data=data, palette={ '0': '#4C72B0', '1': '#DD8452' })  # blue, orange
    plt.title(f'Boxplot of {column} by binary class')
    plt.xlabel('Binary Class')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

