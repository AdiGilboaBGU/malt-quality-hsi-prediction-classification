import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit

#### Best pipeline prediction per seed with groups 

# Function to plot results with regression line and highlight outliers
def plot_with_regression(x, y, title, x_label, y_label, r2_value, outliers_indices=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='darkblue', label='Predictions')  
    if outliers_indices is not None:
        plt.scatter(x[outliers_indices], y[outliers_indices], color='red', s=100, edgecolor='black', label='Outliers')  

    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label='Ideal Fit', linewidth=2)  

    coefficients = np.polyfit(x, y, 1)  
    polynomial = np.poly1d(coefficients)  
    reg_line = polynomial(x)  
    plt.plot(x, reg_line, 'g-', label='Regression Line', linewidth=2)  

    plt.title(f"{title}\n$R^2$: {r2_value:.3f}")  
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function: Successive Projections Algorithm (SPA)
def SPA(X, y, n_features):
    X = np.array(X)
    y = np.array(y).ravel()
    n_samples, n_total_features = X.shape
    selected_features = []
    for _ in range(n_features):
        min_error = np.inf
        selected_feature = None
        for i in range(n_total_features):
            if i in selected_features:
                continue
            current_features = selected_features + [i]
            X_selected = X[:, current_features]
            model = LinearRegression()
            model.fit(X_selected, y)
            y_pred = model.predict(X_selected)
            error = mean_squared_error(y, y_pred)
            if error < min_error:
                min_error = error
                selected_feature = i
        selected_features.append(selected_feature)
    return selected_features

# Function to load data and create group identifiers from Excel
def load_excel_data_and_groups(features_file):

    # Load the data
    X = pd.read_excel(features_file, index_col=0)  # Load features with the first column as the index

    # Extract group identifiers from the index (or another column, if needed)
    groups = X.index.to_series().astype(str).apply(lambda x: x.split('_')[1] if '_' in x else x).astype(int).values

    return X, groups


# Load data
folder_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_prediction\Y_by_seed_prediction.xlsx', index_col=0)
# y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_prediction\expanded_Transformed_Y.xlsx', index_col=0)

# Load selected pipelines file (contains chosen feature selection and model per target variable)
selected_pipelines_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Species\Best_Pipeline_Per_Target_Per_Image.xlsx')
selected_pipelines = selected_pipelines_df.set_index('Y Variable').to_dict(orient='index')

final_results_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Seed_Prediction.xlsx', header=None)
final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}


# Function to calculate mean prediction per group and evaluate model
def predict_mean_per_group_and_evaluate(model, X, y, groups):
#    y = np.array(y).ravel()

    predictions = model.predict(X).flatten()
    df = pd.DataFrame({'Group': groups, 'Predictions': predictions, 'Actual': y})
    group_means = df.groupby('Group').mean().to_dict()['Predictions']
    group_actual_means = df.groupby('Group').mean().to_dict()['Actual']
    group_predictions = np.array([group_means[g] for g in groups])
    group_actual = np.array([group_actual_means[g] for g in groups])
    group_r2 = r2_score(group_actual, group_predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    group_r2 = r2_score(group_actual, group_predictions)

    return mse, rmse, r2, group_r2, predictions


# Define models
models = {
    'LGBM': LGBMRegressor(learning_rate=0.01, n_estimators=200, num_leaves=10, verbose=-1),
    'SVR': SVR(C=10, epsilon=0.01, gamma=0.01),
    'PLSR': PLSRegression(n_components=2),
    'RandomForest': RandomForestRegressor(n_estimators=100),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
}

# Define feature selection techniques
feature_selection_methods = {
    'No Selection': None,  
    'RFE': RFE(estimator=SVR(kernel="linear"), n_features_to_select=10),
    'SelectKBest_f': SelectKBest(score_func=f_regression, k=10),
    'PCA': PCA(n_components=0.95),
    'SPA': SPA
}

# Initialize an empty DataFrame to hold all results
all_results = []

# Process each target variable
for target_var in y_df.columns:
    print(f"\nProcessing Target Variable: {target_var}")
    y = y_df[target_var].values
    data_path = os.path.join(folder_path, final_results[target_var])
    print(f"Using Dataset: {final_results[target_var]}")

    # Load dataset with group information
    X, groups = load_excel_data_and_groups(data_path)
    X = X.select_dtypes(include=[np.number])  
    X.columns = X.columns.astype(str)  

    # Perform GroupShuffleSplit to ensure group separation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=77)

    # Splitting the data into training and testing sets ensuring no group overlap
    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

    # Verify group uniqueness
    print("Groups in Training Set:", np.unique(groups_train))
    print("Groups in Test Set:", np.unique(groups_test))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit & transform only on train set
    X_test = scaler.transform(X_test)  # Transform test set based on train set

    """
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))  # Reshape for a single feature
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    """

    # Get pre-selected feature selection method and model
    selected_method = selected_pipelines[target_var]['Feature Selection Method']
    selected_model = selected_pipelines[target_var]['Model']

    print(f"Selected Feature Selection Method: {selected_method}")
    print(f"Selected Model: {selected_model}")

    # Apply the selected feature selection method
    selector = feature_selection_methods[selected_method]
    if selected_method == 'No Selection':
        X_transformed = X_train
        X_test_transformed = X_test
        selected_features = 'All Features'  # Assign all features if no specific selection is made
    elif selected_method == 'SPA':
        selected_features_indices = selector(X_train, y_train, 10)
        selected_features = X.columns[selected_features_indices].tolist()  # Convert indices to feature names
        X_transformed = X_train[:, selected_features_indices]
        X_test_transformed = X_test[:, selected_features_indices]
    elif selected_method == 'PCA':
        X_transformed = selector.fit_transform(X_train)
        X_test_transformed = selector.transform(X_test)
        selected_features = [f'PC{i+1}' for i in range(selector.n_components_)]  # Naming PCA components
    else:
        X_transformed = selector.fit_transform(X_train, y_train)
        X_test_transformed = selector.transform(X_test)
        if hasattr(selector, 'get_support'):
            selected_features = X.columns[selector.get_support()].tolist()  # Retrieve the names of the selected features if possible
        else:
            selected_features = []  # If the method does not support listing features

    # Train the model only AFTER feature selection
    model = models[selected_model]
#    y_train = np.array(y_train).ravel()

    model.fit(X_transformed, y_train)
    
    # Evaluate performance using the transformed feature set
    train_mse, train_rmse, train_r2, train_group_r2, train_predictions = predict_mean_per_group_and_evaluate(model, X_transformed, y_train, groups_train)
    test_mse, test_rmse, test_r2, test_group_r2, test_predictions = predict_mean_per_group_and_evaluate(model, X_test_transformed, y_test, groups_test)

    best_model_info = {
        'Y Variable': target_var,
        'Feature Selection Method': selected_method,
        'Model': selected_model,
        'Train R2 Score': train_r2,
        'Test R2 Score': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train Group R2': train_group_r2,
        'Test Group R2': test_group_r2,
        'Selected Features': selected_features,
        'Train Predictions': train_predictions.tolist(),
        'Test Predictions': test_predictions.tolist()
    }
    
    all_results.append(best_model_info)

# Save final results to Excel
all_results = pd.DataFrame(all_results)
all_results.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Seed\Best_Results_With_Groups_Per_Seed.xlsx', index=False)
# all_results.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Seed\Best_Results_With_Groups_Per_Seed_Y_Scaled.xlsx', index=False)
# all_results.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Seed\Best_Results_With_Groups_Per_Seed_Y_Transformed.xlsx', index=False)
# all_results.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Seed\Best_Results_With_Groups_Per_Seed_Y_Transformed_Scaled.xlsx', index=False)

print("Results saved successfully!")




####################################################################################
#### Best pipeline prediction average across 100 seeds 
####################################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit

#### Best pipeline prediction per seed with groups 

# Function to plot results with regression line and highlight outliers
def plot_with_regression(x, y, title, x_label, y_label, r2_value, outliers_indices=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='darkblue', label='Predictions')  
    if outliers_indices is not None:
        plt.scatter(x[outliers_indices], y[outliers_indices], color='red', s=100, edgecolor='black', label='Outliers')  

    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label='Ideal Fit', linewidth=2)  

    coefficients = np.polyfit(x, y, 1)  
    polynomial = np.poly1d(coefficients)  
    reg_line = polynomial(x)  
    plt.plot(x, reg_line, 'g-', label='Regression Line', linewidth=2)  

    plt.title(f"{title}\n$R^2$: {r2_value:.3f}")  
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function: Successive Projections Algorithm (SPA)
def SPA(X, y, n_features):
    X = np.array(X)
    y = np.array(y).ravel()
    n_samples, n_total_features = X.shape
    selected_features = []
    for _ in range(n_features):
        min_error = np.inf
        selected_feature = None
        for i in range(n_total_features):
            if i in selected_features:
                continue
            current_features = selected_features + [i]
            X_selected = X[:, current_features]
            model = LinearRegression()
            model.fit(X_selected, y)
            y_pred = model.predict(X_selected)
            error = mean_squared_error(y, y_pred)
            if error < min_error:
                min_error = error
                selected_feature = i
        selected_features.append(selected_feature)
    return selected_features

# Function to load data and create group identifiers from Excel
def load_excel_data_and_groups(features_file):

    # Load the data
    X = pd.read_excel(features_file, index_col=0)  # Load features with the first column as the index

    # Extract group identifiers from the index (or another column, if needed)
    groups = X.index.to_series().astype(str).apply(lambda x: x.split('_')[1] if '_' in x else x).astype(int).values

    return X, groups


# Define a range of seed values for multiple runs
import random
seed_values = random.sample(range(82, 182), 30)


# Load data
folder_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_prediction\Y_by_seed_prediction.xlsx', index_col=0)

# Load selected pipelines file (contains chosen feature selection and model per target variable)
selected_pipelines_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Species\Best_Pipeline_Per_Target_Per_Image.xlsx')
selected_pipelines = selected_pipelines_df.set_index('Y Variable').to_dict(orient='index')

# Load dataset mapping (final_results)
final_results_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Seed_Prediction.xlsx', header=None)
final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}

# Define models
models = {
    'LGBM': LGBMRegressor(learning_rate=0.01, n_estimators=200, num_leaves=10, verbose=-1, random_state=42),
    'SVR': SVR(C=10, epsilon=0.01, gamma=0.01),
    'PLSR': PLSRegression(n_components=2),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Define feature selection techniques
feature_selection_methods = {
    'No Selection': None,
#    'RFE': RFE(estimator=SVR(kernel="linear"), n_features_to_select=10),
    'SelectKBest_f': SelectKBest(score_func=f_regression, k=10),
    'PCA': PCA(n_components=0.95),
    'SPA': SPA
}

all_results = []

# Process each target variable
for target_var in y_df.columns:
    print(f"\nProcessing Target Variable: {target_var}")
    y = y_df[target_var].values
    data_path = os.path.join(folder_path, final_results[target_var])
    print(f"Using Dataset: {final_results[target_var]}")

    # Load dataset with group information
    X, groups = load_excel_data_and_groups(data_path)
    X = X.select_dtypes(include=[np.number])  
    X.columns = X.columns.astype(str)  

    aggregated_results = []
    
    # Iterate over seed values
    for seed in seed_values:
        print(f"Running with seed: {seed}")
        
        # Perform GroupShuffleSplit to ensure group separation
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    
        # Splitting the data into training and testing sets ensuring no group overlap
        for train_idx, test_idx in gss.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]
    
        # Verify group uniqueness
        print("Groups in Training Set:", np.unique(groups_train))
        print("Groups in Test Set:", np.unique(groups_test))
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # Fit & transform only on train set
        X_test = scaler.transform(X_test)  # Transform test set based on train set

   
        # Get pre-selected feature selection method and model
        selected_method = selected_pipelines[target_var]['Feature Selection Method']
        selected_model = selected_pipelines[target_var]['Model']
    
        print(f"Selected Feature Selection Method: {selected_method}")
        print(f"Selected Model: {selected_model}")
    
        # Apply the selected feature selection method
        selector = feature_selection_methods[selected_method]
        if selected_method == 'No Selection':
            X_transformed = X_train
            X_test_transformed = X_test
            selected_features = 'All Features'  # Assign all features if no specific selection is made
        elif selected_method == 'SPA':
            selected_features_indices = selector(X_train, y_train, 10)
            selected_features = X.columns[selected_features_indices].tolist()  # Convert indices to feature names
            X_transformed = X_train[:, selected_features_indices]
            X_test_transformed = X_test[:, selected_features_indices]
        elif selected_method == 'PCA':
            X_transformed = selector.fit_transform(X_train)
            X_test_transformed = selector.transform(X_test)
            selected_features = [f'PC{i+1}' for i in range(selector.n_components_)]  # Naming PCA components
        else:
            X_transformed = selector.fit_transform(X_train, y_train)
            X_test_transformed = selector.transform(X_test)
            if hasattr(selector, 'get_support'):
                selected_features = X.columns[selector.get_support()].tolist()  # Retrieve the names of the selected features if possible
            else:
                selected_features = []  # If the method does not support listing features
    
        # Train the model only AFTER feature selection
        model = models[selected_model]
    #    y_train = np.array(y_train).ravel()
    
        model.fit(X_transformed, y_train)

        predictions = model.predict(X_test_transformed)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Identify and remove outliers
        residuals = y_test - predictions
        large_residual_indices = np.argsort(np.abs(residuals))[-4:]

        # Compute statistics after removing outliers
        y_test_filtered = np.delete(y_test, large_residual_indices)
        predictions_filtered = np.delete(predictions, large_residual_indices)
        r2_filtered = r2_score(y_test_filtered, predictions_filtered)
        rmse_filtered = np.sqrt(mean_squared_error(y_test_filtered, predictions_filtered))

        aggregated_results.append([target_var, selected_method, selected_model, r2, r2_filtered, rmse, rmse_filtered])

    # Create a DataFrame from the results
#    df_results = pd.DataFrame(aggregated_results, columns=['Y Variable', 'Feature Selection Method', 'Model', 'R2 Score', 'R2 Score Filtered', 'RMSE', 'RMSE Filtered'])

    df_results = pd.DataFrame(aggregated_results, columns=['Y Variable', 'Feature Selection Method', 'Model', 'R2 Score', 'R2 Score Filtered', 'RMSE', 'RMSE Filtered'])
    
    # Compute statistics for R2 Score and RMSE before and after removing outliers
    stats_df = df_results.groupby(['Y Variable', 'Feature Selection Method', 'Model']).agg({
        'R2 Score': ['mean', 'std', 'min', 'max'],
        'R2 Score Filtered': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'RMSE Filtered': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # Flatten MultiIndex column names
    stats_df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in stats_df.columns]
    print("Columns after flattening:", stats_df.columns.tolist())
    # Select the best model based on the highest filtered R² score
#    best_model = df_results.loc[df_results.groupby("Y Variable")["R2 Score Filtered"].idxmax()]

    # Select the best model based on the highest filtered R² score
    best_model = stats_df.loc[stats_df.groupby("Y Variable")["R2 Score Filtered mean"].idxmax().values]
    
    # Print the best model for the target variable
    print(f"\nBest model for {target_var}:")
    print(best_model.to_string(index=False))


    # Print the best model for the target variable
    print(f"\nBest model for {target_var}:")
    print(best_model.to_string(index=False))

    # Generate plots for the best model
    best_row = best_model.iloc[0]
    # Generate plots for the best model
    plot_with_regression(y_test, predictions, 
                         f"{target_var} - {best_row['Model']} (Before Outliers)", 
                         'Actual', 'Predicted', best_row['R2 Score mean'], 
                         outliers_indices=large_residual_indices)
    
    plot_with_regression(y_test_filtered, predictions_filtered, 
                         f"{target_var} - {best_row['Model']} (After Outliers)", 
                         'Actual', 'Predicted', best_row['R2 Score Filtered mean'])

    all_results.append(best_model)

# Save the best model results to an Excel file
df_best_models = pd.concat(all_results, ignore_index=True)
df_best_models.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Prediction_Per_Seed\Best_Pipeline_Per_Seed_Per_Image_100_Set_Seed.xlsx', index=False)

print(stats_df.columns)

