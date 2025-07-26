import pandas as pd
import numpy as np


#### EXPAND Y PREDICTION TO SEED LEVEL
##########################################################################################################################
##########################################################################################################################


import pandas as pd

# Load the data files
data_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y.xlsx'

data = pd.read_excel(data_path)

# Function to replicate each value in the DataFrame
def replicate_rows(df, default_times=64, exception_row=33, exception_times=62):
    result = pd.DataFrame()  # Initialize an empty DataFrame to store results
    for index, row in df.iterrows():
        times = exception_times if row['Image_Name'] == exception_row else default_times
        result = pd.concat([result] + [row.to_frame().T] * times, ignore_index=True)
    return result

# Apply the replication function to binary and ternary data
expanded_prediction_data = replicate_rows(data)

# Save the expanded binary and ternary data to new Excel files
expanded_prediction_data.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\expanded_prediction_data.xlsx', index=False)
