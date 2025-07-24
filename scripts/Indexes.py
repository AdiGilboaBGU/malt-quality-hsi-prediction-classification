import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


############################################################################################################

# Step 1: Load the data
X = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\average_reflectances_with_erosion_1_NONE.xlsx', index_col=0)
X = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\average_reflectances_with_erosion1.xlsx', index_col=0)

# Get the number of columns dynamically
num_columns = X.shape[1]

# Step 2: Calculate the indexes for each unique pair of columns
indexes_df = pd.DataFrame()

for i in range(num_columns):
    for j in range(i+1, num_columns):  # Start j from i+1 to avoid repeating pairs
        col1 = X.iloc[:, i]
        col2 = X.iloc[:, j]

        # Calculate the index: (col1 - col2) / (col1 + col2)
        index = np.abs((col1 - col2) / (col1 + col2))

        # Naming the index based on column numbers
        index_name = f'Index_{i+1}-{j+1}'
        indexes_df[index_name] = index
        
## Too big to export to excel

# The target variable
y =  pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\Y.xlsx')
y = y.apply(pd.to_numeric)  # Converts data to numeric, setting errors to NaN
y = y / 100 
# Turn Y into series variable
y_series = y.iloc[:, 0]

# Reset the indexes to match it to Y 
indexes_df = indexes_df.reset_index(drop=True)

# Calculate the correlation between each feature and the target
correlation_with_target = indexes_df.corrwith(y_series)

# Filter features by high correlation with the target
# Select the top 204 features with the highest correlation
top_features = correlation_with_target.abs().nlargest(204).index.tolist()

# Save the correlations of the selected features
selected_feature_correlations = correlation_with_target[top_features]

# Create a new DataFrame with the selected features and the target
selected_df = indexes_df[top_features]

# Save the reduced DataFrame to an Excel file
selected_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\selected_indexes_high_corr.xlsx', index=False)

print("Selected features saved to Excel.")



############################################################################################################
#### PER SEED
############################################################################################################

# Step 1: Load the data
X = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\X_per_seed.xlsx', index_col=0)
# X = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\average_reflectances_with_erosion1.xlsx', index_col=0)

# Get the number of columns dynamically
num_columns = X.shape[1]

# Step 2: Calculate the indexes for each unique pair of columns
indexes_df = pd.DataFrame()

for i in range(num_columns):
    for j in range(i+1, num_columns):  # Start j from i+1 to avoid repeating pairs
        col1 = X.iloc[:, i]
        col2 = X.iloc[:, j]

        # Calculate the index: (col1 - col2) / (col1 + col2)
        index = np.abs((col1 - col2) / (col1 + col2))

        # Naming the index based on column numbers
        index_name = f'Index_{i+1}-{j+1}'
        indexes_df[index_name] = index
        
## Too big to export to excel

# The target variable
# y =  pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\testing_datasets\Y.xlsx')
# y = y.apply(pd.to_numeric)  # Converts data to numeric, setting errors to NaN
# y = y / 100 
# Turn Y into series variable
# y_series = y.iloc[:, 0]

# Reset the indexes to match it to Y 
indexes_df = indexes_df.reset_index(drop=True)

# Step 3: Select top 204 features based on variance
variances = indexes_df.var()
# Sort the variances and select the top 204 columns
top_features = variances.sort_values(ascending=False).head(204).index
# Create a new DataFrame with the selected features and the target
selected_df = indexes_df[top_features]


# Save the reduced DataFrame to an Excel file
selected_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files\X_per_seed_indexes.xlsx', index=False)

print("Selected features saved to Excel.")
