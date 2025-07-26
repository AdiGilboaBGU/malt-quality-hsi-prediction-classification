import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### Descriptive statistics of X features
########################################################

#--------------------------------------------------------
# Summary statistics table of 10 random features
#--------------------------------------------------------

import pandas as pd
import numpy as np

# Load the reflectance data
# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_species\average_reflectances_with_erosion_1_NONE.xlsx', index_col=0))
reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_seed\X_by_seed.xlsx', index_col=0))

# reflectance_df = reflectance_df.set_index('Labels')

# Full list of 204 wavelengths (index 0 to 203)
wavelengths = [
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.62, 417.52, 420.40, 423.29,
    426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
    455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
    484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
    513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
    542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
    572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
    601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
    631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
    660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
    690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
    720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
    750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
    780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
    810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
    841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
    871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
    902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
    932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
    963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
    994.31, 997.40, 1000.49, 1003.58
]

# Convert to NumPy for slicing
wavelengths = np.array(wavelengths)

# Get wavelengths between 420 and 980
valid_indices = np.where((wavelengths >= 420) & (wavelengths <= 980))[0]

# Choose 10 evenly spaced indices in this range
selected_indices = np.linspace(valid_indices[0], valid_indices[-1], 10, dtype=int)

# Extract corresponding wavelengths
selected_wavelengths = wavelengths[selected_indices]

# Extract the relevant columns from the dataframe
selected_columns = reflectance_df.columns[selected_indices]
selected_df = reflectance_df[selected_columns]

# Rename columns to wavelength values (e.g., '420.4 nm')
selected_df.columns = [f"{w:.2f} nm" for w in selected_wavelengths]

# Compute and print descriptive statistics
summary_stats = pd.DataFrame({
    'Mean': selected_df.mean(),
    'Median': selected_df.median(),
    'Std': selected_df.std(),
    'Min': selected_df.min(),
    'Max': selected_df.max(),
    'Skewness': selected_df.skew(),
    'Kurtosis': selected_df.kurt()
})

# Print summary table
print(summary_stats.round(3))



#--------------------------------------------------------
# Of all features
#--------------------------------------------------------


import pandas as pd
import numpy as np

# Load the reflectance data
reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_species\average_reflectances_with_erosion_1_NONE.xlsx'))
reflectance_df = reflectance_df.set_index('Labels')

# Define the full list of 204 wavelengths
wavelengths = [
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.62, 417.52, 420.40, 423.29,
    426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
    455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
    484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
    513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
    542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
    572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
    601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
    631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
    660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
    690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
    720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
    750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
    780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
    810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
    841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
    871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
    902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
    932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
    963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
    994.31, 997.40, 1000.49, 1003.58
]

# Set the column names to wavelength values
reflectance_df.columns = [f"{w:.2f} nm" for w in wavelengths]

# Compute summary statistics for all 204 wavelengths
summary_stats = pd.DataFrame({
    'Mean': reflectance_df.mean(),
    'Std': reflectance_df.std(),
    'Min': reflectance_df.min(),
    'Max': reflectance_df.max(),
    'Range': reflectance_df.max() - reflectance_df.min(),
    'Skewness': reflectance_df.skew(),
    'Kurtosis': reflectance_df.kurt()
})

# Round and save
summary_stats = summary_stats.round(3)
# Optional: also show it on screen
print(summary_stats)

summary_stats.to_excel(os.path.join('..', '..', 'datasets', 'summary_stats_X_all_features_named.xlsx'))




#--------------------------------------------------------
# Spectral signature
#--------------------------------------------------------

# Load the reflectance data

reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_species\average_reflectances_with_erosion_1_NONE.xlsx', index_col=0))
# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_species\average_reflectances_with_erosion_1_SG_SD.xlsx', index_col=0))
# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_species\average_reflectances_with_erosion_1_SG_MSC.xlsx', index_col=0))
# flectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_species\average_reflectances_with_erosion_1_SG_SNV.xlsx', index_col=0))

# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_seed\X_by_seed.xlsx', index_col=0))
# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_seed\X_by_seed_SG_SD.xlsx', index_col=0))
# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_seed\X_by_seed_SG_MSC.xlsx', index_col=0))
# reflectance_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'X_by_seed\X_by_seed_SG_SNV.xlsx', index_col=0))



# Create wavelength list (length = 204)
wavelengths = [
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.62, 417.52, 420.40, 423.29,
    426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
    455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
    484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
    513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
    542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
    572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
    601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
    631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
    660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
    690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
    720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
    750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
    780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
    810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
    841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
    871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
    902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
    932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
    963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
    994.31, 997.40, 1000.49, 1003.58
]

# Convert DataFrame to numpy for plotting
spectral_data = reflectance_df.to_numpy()

# Plot spectral signatures
plt.figure(figsize=(12, 6))
for i in range(spectral_data.shape[0]):
    plt.plot(wavelengths, spectral_data[i], linewidth=0.8)

plt.title("Spectral signatures of barley varieties")
# plt.title("Spectral signatures of barley varieties after SG_SD")
# plt.title("Spectral signatures of barley varieties after SG_MSC")
# plt.title("Spectral signatures of barley varieties after SG_SNV")

# plt.title("Spectral signatures of barley seeds")
# plt.title("Spectral signatures of barley seeds after SG_SD")
# plt.title("Spectral signatures of barley seeds after SG_MSC")
# plt.title("Spectral signatures of barley seeds after SG_SNV")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.grid(True)
plt.tight_layout()
plt.show()



