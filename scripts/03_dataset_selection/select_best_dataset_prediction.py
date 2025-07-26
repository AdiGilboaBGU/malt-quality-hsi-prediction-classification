import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
import numpy as np

##############################################################################################################################
#### COMPARE DATASET PER SPECIES FOR PREDICTION
##############################################################################################################################
##############################################################################################################################

def rmse_cv(model, X, y):
    """Calculate the cross-validated RMSE"""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

def train_models(X_train, X_test, y_train, y_test):
    """Train different models and return their RMSECV and R2 metrics."""
    models = {
        'SVR': SVR(),
        'XGB': XGBRegressor(),
        'PLSR': PLSRegression(n_components=5)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse_cv_score = rmse_cv(model, X_train, y_train)
        r2_score_val = r2_score(y_test, predictions)
        results[name] = {'RMSECV': rmse_cv_score, 'R2': r2_score_val}
    return results

# Reading the Excel files
# Path to the folder containing the Excel files
datasets_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_species'
# geo_params_df = pd.read_excel(os.path.join(datasets_path, 'geometric_parameters.xlsx'))
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_prediction\Y.xlsx', index_col=0)

# Initialize a dictionary to store the best results for each Y variable
final_results = {} 

# Process each Y variable independently
for target_var in y_df.columns:
    print(f"Processing target variable: {target_var}")
    y = y_df[target_var]
#    y_scaler = StandardScaler()
#    y_scaled = y_scaler.fit_transform(y_series.values.reshape(-1, 1)).ravel()

    # Initialize results summary and best results tracking for this variable
    results_summary = []
    best_results = None
    best_dataset = None

    for file_name in os.listdir(datasets_path):
        if file_name.endswith('.xlsx') and file_name not in ['geometric_parameters.xlsx'] and not file_name.startswith('~$'):
            reflectance_df = pd.read_excel(os.path.join(datasets_path, file_name))
#            combined_df = pd.concat([reflectance_df, geo_params_df], axis=1)
            combined_df = reflectance_df.drop(columns=['Labels'])
            # Redefining column names to be sequential numbers
            combined_df.columns = range(combined_df.shape[1])

            X = combined_df

            # Split data into Train and Test sets BEFORE applying StandardScaler
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)
            
            # Apply StandardScaler correctly
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on Train set
            X_test_scaled = scaler.transform(X_test)  # Only transform Test set (no fit!)
            
            # Replace the original sets with the scaled versions
            X_train = X_train_scaled
            X_test = X_test_scaled


            results = train_models(X_train, X_test, y_train, y_test)
            results['dataset'] = file_name
            results_summary.append(results)
            
            # Initialize best_results on first iteration
            if best_results is None:
                best_results = results
                best_dataset = file_name

            # Update best results dynamically
            better_models_count = sum(
                1 for model in results if model != 'dataset' and results[model]['RMSECV'] < best_results[model]['RMSECV']
            )
            
            print(f"Processing {file_name}: better_models_count = {better_models_count}")
            
            if better_models_count >= 2:  # The dataset must be better in at least two models to be considered the best
                best_dataset = file_name  # Update the best dataset to the current file
                best_results = results  # Update the best results to the current results

    # Save the best results for this Y variable in the final_results dictionary
    final_results[target_var] = {
        'Best Dataset': best_dataset,
        'Best Results': best_results
    }  # Added line to store results
   
final_results = pd.DataFrame(final_results)
final_results.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Species_With_Geometric.xlsx', index=False)

###############################################################

import pandas as pd

# Settings to display all rows and columns in a pandas DataFrame
pd.set_option('display.max_rows', None)  # This setting allows displaying all rows in a DataFrame
pd.set_option('display.max_columns', None)  # This setting allows displaying all columns in a DataFrame
pd.set_option('display.width', 1000)  # Increases the display width to accommodate more text
pd.set_option('display.max_colwidth', None)  # Allows column content to be displayed without truncation

min_rmsecv_per_y = []
for y_var, results in final_results.items():
    print(f"Y Variable: {y_var}, Results: {results}")  # Debugging output
    best_dataset = results['Best Dataset']
    best_results = results['Best Results']
    min_rmsecv = float('inf')
    best_model_name = None
    
    for model_name, model_results in best_results.items():
        print(f"Model Name: {model_name}, Model Results: {model_results}")  # More debugging output
        if 'RMSECV' in model_results and model_results['RMSECV'] < min_rmsecv:
            min_rmsecv = model_results['RMSECV']
            best_model_name = model_name
    
    min_rmsecv_per_y.append({
        'Y Variable': y_var,
        'Min RMSECV': min_rmsecv,
        'Best Dataset': best_dataset,
        'Best Model': best_model_name
    })

# Convert to DataFrame for display
min_rmsecv_df = pd.DataFrame(min_rmsecv_per_y)
min_rmsecv_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\min_rmsecv_df.xlsx', index=False)
print(min_rmsecv_df)

#--------------------------------------------------------------------------------

max_r2_per_y = []
for y_var, results in final_results.items():
    print(f"Y Variable: {y_var}, Results: {results}")  # Debugging output
    best_dataset = results['Best Dataset']
    best_results = results['Best Results']
    max_r2 = -float('inf')  # Initialize to negative infinity to find maximum
    best_model_name = None
    
    for model_name, model_results in best_results.items():
        print(f"Model Name: {model_name}, Model Results: {model_results}")  # More debugging output
        if 'R2' in model_results and model_results['R2'] > max_r2:
            max_r2 = model_results['R2']
            best_model_name = model_name
    
    max_r2_per_y.append({
        'Y Variable': y_var,
        'Max R2': max_r2,
        'Best Dataset': best_dataset,
        'Best Model': best_model_name
    })

# Convert to DataFrame for display
max_r2_df = pd.DataFrame(max_r2_per_y)
max_r2_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\max_r2_df.xlsx', index=False)
print(max_r2_df)

##############################################################################################################################
#### COMPARE DATASET PER SEED FOR PREDICTION
##############################################################################################################################
##############################################################################################################################

def rmse_cv(model, X, y):
    """Calculate the cross-validated RMSE"""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

def train_models(X_train, X_test, y_train, y_test):
    """Train different models and return their RMSECV and R2 metrics."""
    models = {
        'SVR': SVR(),
        'XGB': XGBRegressor(),
        'PLSR': PLSRegression(n_components=5)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse_cv_score = rmse_cv(model, X_train, y_train)
        r2_score_val = r2_score(y_test, predictions)
        results[name] = {'RMSECV': rmse_cv_score, 'R2': r2_score_val}
    return results

# Reading the Excel files
# Path to the folder containing the Excel files
datasets_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
# geo_params_df = pd.read_excel(os.path.join(datasets_path, 'geometric_parameters.xlsx'))
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_prediction\Y_by_seed_prediction.xlsx', index_col=0)

# Initialize a dictionary to store the best results for each Y variable
final_results = {} 

# Process each Y variable independently
for target_var in y_df.columns:
    print(f"Processing target variable: {target_var}")
    y = y_df[target_var]
#    y_scaler = StandardScaler()
#    y_scaled = y_scaler.fit_transform(y_series.values.reshape(-1, 1)).ravel()

    # Initialize results summary and best results tracking for this variable
    results_summary = []
    best_results = None
    best_dataset = None

    for file_name in os.listdir(datasets_path):
        if file_name.endswith('.xlsx') and file_name not in ['geometric_parameters.xlsx'] and not file_name.startswith('~$'):
            combined_df = pd.read_excel(os.path.join(datasets_path, file_name), index_col=0)
#            combined_df = pd.concat([reflectance_df, geo_params_df], axis=1)
#            combined_df = reflectance_df.drop(columns=['Labels'])
            # Redefining column names to be sequential numbers
            combined_df.columns = range(combined_df.shape[1])

            X = combined_df

            # Split data into Train and Test sets BEFORE applying StandardScaler
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)
            
            # Apply StandardScaler correctly
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on Train set
            X_test_scaled = scaler.transform(X_test)  # Only transform Test set (no fit!)
            
            # Replace the original sets with the scaled versions
            X_train = X_train_scaled
            X_test = X_test_scaled

            results = train_models(X_train, X_test, y_train, y_test)
            results['dataset'] = file_name
            results_summary.append(results)
            
            # Initialize best_results on first iteration
            if best_results is None:
                best_results = results
                best_dataset = file_name

            # Update best results dynamically
            better_models_count = sum(
                1 for model in results if model != 'dataset' and results[model]['RMSECV'] < best_results[model]['RMSECV']
            )
            
            print(f"Processing {file_name}: better_models_count = {better_models_count}")
            
            if better_models_count >= 2:  # The dataset must be better in at least two models to be considered the best
                best_dataset = file_name  # Update the best dataset to the current file
                best_results = results  # Update the best results to the current results

    # Save the best results for this Y variable in the final_results dictionary
    final_results[target_var] = {
        'Best Dataset': best_dataset,
        'Best Results': best_results
    }  # Added line to store results
   
final_results = pd.DataFrame(final_results)
final_results.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Seed_Without_Geometric.xlsx', index=False)


###############################################################

import pandas as pd

# Settings to display all rows and columns in a pandas DataFrame
pd.set_option('display.max_rows', None)  # This setting allows displaying all rows in a DataFrame
pd.set_option('display.max_columns', None)  # This setting allows displaying all columns in a DataFrame
pd.set_option('display.width', 1000)  # Increases the display width to accommodate more text
pd.set_option('display.max_colwidth', None)  # Allows column content to be displayed without truncation

min_rmsecv_per_y = []
for y_var, results in final_results.items():
    print(f"Y Variable: {y_var}, Results: {results}")  # Debugging output
    best_dataset = results['Best Dataset']
    best_results = results['Best Results']
    min_rmsecv = float('inf')
    best_model_name = None
    
    for model_name, model_results in best_results.items():
        print(f"Model Name: {model_name}, Model Results: {model_results}")  # More debugging output
        if 'RMSECV' in model_results and model_results['RMSECV'] < min_rmsecv:
            min_rmsecv = model_results['RMSECV']
            best_model_name = model_name
    
    min_rmsecv_per_y.append({
        'Y Variable': y_var,
        'Min RMSECV': min_rmsecv,
        'Best Dataset': best_dataset,
        'Best Model': best_model_name
    })

# Convert to DataFrame for display
min_rmsecv_df = pd.DataFrame(min_rmsecv_per_y)
min_rmsecv_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\min_rmsecv_df.xlsx', index=False)
print(min_rmsecv_df)

#--------------------------------------------------------------------------------

max_r2_per_y = []
for y_var, results in final_results.items():
    print(f"Y Variable: {y_var}, Results: {results}")  # Debugging output
    best_dataset = results['Best Dataset']
    best_results = results['Best Results']
    max_r2 = -float('inf')  # Initialize to negative infinity to find maximum
    best_model_name = None
    
    for model_name, model_results in best_results.items():
        print(f"Model Name: {model_name}, Model Results: {model_results}")  # More debugging output
        if 'R2' in model_results and model_results['R2'] > max_r2:
            max_r2 = model_results['R2']
            best_model_name = model_name
    
    max_r2_per_y.append({
        'Y Variable': y_var,
        'Max R2': max_r2,
        'Best Dataset': best_dataset,
        'Best Model': best_model_name
    })

# Convert to DataFrame for display
max_r2_df = pd.DataFrame(max_r2_per_y)
max_r2_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\max_r2_df.xlsx', index=False)
print(max_r2_df)