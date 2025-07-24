import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, shapiro, boxcox, skew

# Load the data
labels_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y.xlsx'
y_labels_df = pd.read_excel(labels_file, index_col=0)

# Initialize a list to store results
results = []

# Loop through each column to perform analysis
# Loop through each column to perform analysis
for column in y_labels_df.columns:
    # Plot histogram
    plt.figure(figsize=(10, 6))
    data = y_labels_df[column].dropna()  # Drop NA values for valid statistical analysis
    plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Calculate and plot mean line
    mean_value = data.mean()
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_value*1.05, plt.ylim()[1]*0.9, f'Mean: {mean_value:.2f}', color = 'red')

    # Calculate kurtosis and add to plot
    kurt = kurtosis(data, fisher=True)
    plt.text(mean_value*1.05, plt.ylim()[1]*0.85, f'Kurtosis: {kurt:.2f}', color = 'green')

    plt.grid(False)
    plt.show()
    
    # Continue with your existing calculations and append to results as before
    variance = data.var()
    std_dev = np.sqrt(variance)
    cv = std_dev / mean_value if mean_value != 0 else np.nan
    unique_values = data.nunique()
    
    # Calculate mean
    mean_value = y_labels_df[column].mean()
    
    # Calculate standard deviation
    std_dev = np.sqrt(variance)
    
    # Calculate Coefficient of Variation (CV)
    cv = std_dev / mean_value if mean_value != 0 else np.nan
    
    # Assess continuity
    classification = "Continuous" if unique_values > 10 else "Categorical (Discrete) - Suitable for Classification"
    
    # Determine classification based on CV threshold
    if cv < 0.1:
        cv_classification = "Likely Classification"
    elif 0.1 <= cv < 0.5:
        cv_classification = "Borderline - Check Both"
    else:
        cv_classification = "Likely Regression"
    
    # Calculate entropy
    value_counts = y_labels_df[column].value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts))

    # Calculate kurtosis
    kurt = kurtosis(y_labels_df[column], fisher=True)

    # Calculate skewness
    skewness = skew(y_labels_df[column])

    # Normality test (Shapiro-Wilk Test)
    shapiro_test_stat, shapiro_p_value = shapiro(y_labels_df[column].dropna()) if len(y_labels_df[column]) < 5000 else (np.nan, np.nan)

    # Normality assessment
    normality = "Normally Distributed" if shapiro_p_value > 0.05 else "Not Normally Distributed"

    # Suggest transformation
    transformation_suggestion = "No Transformation Needed"
    
    if shapiro_p_value < 0.05:  # If not normally distributed
        if all(y_labels_df[column] > 0):  # Box-Cox requires all positive values
            try:
                _, best_lambda = boxcox(y_labels_df[column].dropna())  
                transformation_suggestion = f"Box-Cox (λ={best_lambda:.2f})"
            except:
                transformation_suggestion = "Box-Cox Not Feasible"
        elif cv > 0.7 or kurt > 3:
            transformation_suggestion = "Log Transformation"
        elif 0.3 < cv <= 0.7 or (1 < kurt <= 3):
            transformation_suggestion = "Square Root Transformation"
        elif 0.3 < cv <= 0.7 or (0 < kurt <= 3):
            transformation_suggestion = "Cube Root Transformation"

    # Store results
    results.append({
        "Variable": column,
        "Variance": variance,
        "Unique Values": unique_values,
        "Mean": mean_value,
        "Standard Deviation": std_dev,
        "CV": cv,
        "Entropy": entropy,
        "Kurtosis": kurt,
        "Skewness": skewness,  # Added skewness to the results
        "Shapiro-Wilk p-value": shapiro_p_value,
        "Normality": normality,
        "Continuity Assessment": classification,
        "CV Classification": cv_classification,
        "Recommended Transformation": transformation_suggestion
    })

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame(results)

# Save to Excel file
output_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_analysis_results_with_transformation.xlsx'
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
