import pandas as pd
import numpy as np
from scipy.stats import boxcox, shapiro

# Load the data
labels_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y.xlsx'
y_labels_df = pd.read_excel(labels_file, index_col=0)

# Load transformation recommendations
transformations_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y_analysis_results_with_transformation.xlsx')
transformations_map = transformations_df.set_index('Variable')['Recommended Transformation'].to_dict()

# Initialize a DataFrame to store transformed data
transformed_data = pd.DataFrame(index=y_labels_df.index)

# Apply transformations and test normality
normality_results = []

for column in y_labels_df.columns:
    data = y_labels_df[column].dropna()  # Remove NaN values
    transformation = transformations_map[column]
    transformed = data.copy()

    # Apply the recommended transformation
    if "Box-Cox" in transformation:
        lambda_value = float(transformation.split('=')[1].replace(')', ''))
        transformed, _ = boxcox(data + (data <= 0) * 1e-6)  # Shift data if any non-positive values
    elif "Log" in transformation:
        transformed = np.log(data + (data <= 0) * 1e-6)
    elif "Square Root" in transformation:
        transformed = np.sqrt(data)
    elif "Cube Root" in transformation:
        transformed = np.cbrt(data)

    transformed_data[column] = transformed

    # Test normality of the transformed data
    stat, p_value = shapiro(transformed)
    is_normal = "Yes" if p_value > 0.05 else "No"
    normality_results.append(f"{column}: Normal distribution after transformation? {is_normal}")

    # Print normality result
    print(f"{column} - Normal after transformation? {is_normal}")

# Save transformed data to a new Excel file
output_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Transformed_Y.xlsx'
transformed_data.to_excel(output_file)

# Print the saved file path
print(f"Transformed data saved to {output_file}")

# Print normality results
for result in normality_results:
    print(result)

