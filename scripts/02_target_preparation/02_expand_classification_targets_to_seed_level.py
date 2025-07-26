import pandas as pd
import numpy as np

#### EXPAND Y BINARY AND TERNARY CLASSIFICATION TO SEED LEVEL
##########################################################################################################################
##########################################################################################################################

import pandas as pd

# Load the data files
binary_data_path = r'C:\Users\USER\Desktop\excel_files\binary_data.xlsx'
ternary_data_path = r'C:\Users\USER\Desktop\excel_files\ternary_data.xlsx'

binary_data = pd.read_excel(binary_data_path)
ternary_data = pd.read_excel(ternary_data_path)

# Function to replicate each value in the DataFrame
def replicate_rows(df, default_times=64, exception_row=33, exception_times=62):
    result = pd.DataFrame()  # Initialize an empty DataFrame to store results
    for index, row in df.iterrows():
        times = exception_times if row['Image_Name'] == exception_row else default_times
        result = pd.concat([result] + [row.to_frame().T] * times, ignore_index=True)
    return result

# Apply the replication function to binary and ternary data
expanded_binary_data = replicate_rows(binary_data)
expanded_ternary_data = replicate_rows(ternary_data)

# Save the expanded binary and ternary data to new Excel files
expanded_binary_data.to_excel(r'C:\Users\USER\Desktop\excel_files\expanded_binary_data.xlsx', index=False)
expanded_ternary_data.to_excel(r'C:\Users\USER\Desktop\excel_files\expanded_ternary_data.xlsx', index=False)
