import pandas as pd
import numpy as np
import os

#### CONVERT Y TO BINARY AND TERNARY CLASSIFICATION
##########################################################################################################################
##########################################################################################################################


# Load the data
file_path = os.path.join('..', '..', 'datasets', 'Y_by_species_prediction\Y.xlsx')
data = pd.read_excel(file_path)

# Set 'Image_Name' as index
data.set_index('Image_Name', inplace=True)

# Prepare data containers
binary_data = pd.DataFrame()
ternary_data = pd.DataFrame()
binary_stats = {}
ternary_stats = {}
binary_range_stats = {}
ternary_range_stats = {}

# Process each column
for column in data.columns:
    values = data[column]
    sorted_values = np.sort(values.unique())  # Sort unique values to avoid splitting identical values

    # Binary Classification
    best_binary_index = None
    min_difference = float('inf')
    for i in range(1, len(sorted_values)):
        lower_count = (values <= sorted_values[i - 1]).sum()
        upper_count = (values > sorted_values[i - 1]).sum()
        if abs(lower_count - upper_count) < min_difference:
            min_difference = abs(lower_count - upper_count)
            best_binary_index = i - 1

    binary_threshold = sorted_values[best_binary_index]
    binary_data[column] = (values > binary_threshold).astype(int)
    binary_stats[column] = binary_data[column].value_counts().to_dict()
    binary_range_stats[column] = {
        '0': (values[binary_data[column] == 0].min(), values[binary_data[column] == 0].max()),
        '1': (values[binary_data[column] == 1].min(), values[binary_data[column] == 1].max())
    }

    # Ternary Classification
    best_ternary_indices = (None, None)
    min_difference = float('inf')
    for i in range(1, len(sorted_values) - 1):
        for j in range(i + 1, len(sorted_values)):
            first_count = (values <= sorted_values[i - 1]).sum()
            second_count = ((values > sorted_values[i - 1]) & (values <= sorted_values[j - 1])).sum()
            third_count = (values > sorted_values[j - 1]).sum()
            max_diff = max(abs(first_count - second_count), abs(second_count - third_count), abs(first_count - third_count))
            if max_diff < min_difference:
                min_difference = max_diff
                best_ternary_indices = (i - 1, j - 1)

    first_ternary_threshold, second_ternary_threshold = sorted_values[best_ternary_indices[0]], sorted_values[best_ternary_indices[1]]
    bins = [-np.inf, first_ternary_threshold, second_ternary_threshold, np.inf]
    ternary_data[column] = pd.cut(values, bins=bins, labels=[0, 1, 2], include_lowest=True).astype(int)
    ternary_stats[column] = ternary_data[column].value_counts().to_dict()
    ternary_range_stats[column] = {
        '0': (values[ternary_data[column] == 0].min(), values[ternary_data[column] == 0].max()),
        '1': (values[ternary_data[column] == 1].min(), values[ternary_data[column] == 1].max()),
        '2': (values[ternary_data[column] == 2].min(), values[ternary_data[column] == 2].max())
    }

# Save the processed data to Excel files
binary_data.to_excel(os.path.join('..', '..', 'datasets', 'binary_data.xlsx'), index=True)
ternary_data.to_excel(os.path.join('..', '..', 'datasets', 'ternary_data.xlsx'), index=True)

# Print out the statistics for each variable in sorted order by class number
print("Binary Classification Stats:")
for column in binary_stats:
    print(f"{column}:")
    for key in sorted(binary_stats[column].keys(), key=int):  # Ensure keys are sorted numerically
        print(f"  Class {key}: {binary_stats[column][key]} entries")
        if str(key) in binary_range_stats[column]:  # Convert key to string for safe dictionary access
            print(f"    Range: {binary_range_stats[column][str(key)]}")

print("\nTernary Classification Stats:")
for column in ternary_stats:
    print(f"{column}:")
    for key in sorted(ternary_stats[column].keys(), key=int):  # Ensure keys are sorted numerically
        print(f"  Class {key}: {ternary_stats[column][key]} entries")
        if str(key) in ternary_range_stats[column]:  # Convert key to string for safe dictionary access
            print(f"    Range: {ternary_range_stats[column][str(key)]}")








