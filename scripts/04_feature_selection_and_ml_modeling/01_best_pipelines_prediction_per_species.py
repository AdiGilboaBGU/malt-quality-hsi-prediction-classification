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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Function to plot results with regression line and highlight outliers
def plot_with_regression(x, y, title, x_label, y_label, r2_value, outliers_indices=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='darkblue', label='Predictions')  # Change points to dark blue
    if outliers_indices is not None:
        plt.scatter(x[outliers_indices], y[outliers_indices], color='red', s=100, edgecolor='black', label='Outliers')  # Highlight outliers in red
    
    # Ideal fit line in black
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label='Ideal Fit', linewidth=2)  # Ideal fit line in black
    
    # Calculate and plot regression line
    coefficients = np.polyfit(x, y, 1)  # Fit a linear model
    polynomial = np.poly1d(coefficients)  # Create a polynomial function
    reg_line = polynomial(x)  # Calculate y values based on the polynomial function
    plt.plot(x, reg_line, 'g-', label='Regression Line', linewidth=2)  # Regression line in green

    plt.title(f"{title}\n$R^2$: {r2_value:.3f}")  # Update title with R^2
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_predicted_vs_actual(y_true, y_pred, title_param):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    ax = sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, s=50, color="steelblue", edgecolor="w", label='Predictions')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', label='Ideal Fit')

    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m * y_true + b, color='green', label='Regression Line')

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs. Actual – {title_param}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residual_distribution(y_true, y_pred, title_param):
    import seaborn as sns
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="tomato", bins=30)
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Residual Distribution – {title_param}")
    plt.xlabel("Residuals")
    plt.ylabel("Count")
    plt.tight_layout()
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

# Load data
folder_path = os.path.join('..', '..', 'datasets', 'X_by_species')
# y_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'Y_by_species_prediction\Y.xlsx', index_col=0))
y_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'Y_by_species_prediction\Transformed_Y.xlsx', index_col=0))

final_results_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Species_Prediction.xlsx', header=None))
final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}



# Define models
models = {
    'LGBM': LGBMRegressor(learning_rate=0.01, n_estimators=200, num_leaves=10, verbose = -1),
    'SVR': SVR(C=10, epsilon=0.01, gamma=0.01),
    'PLSR': PLSRegression(n_components=2),
    'RandomForest': RandomForestRegressor(n_estimators=100),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
}

# Define feature selection techniques
feature_selection_methods = {
    'No Selection': None,  #  Option without feature selection
#    'RFE': RFE(estimator=SVR(kernel="linear"), n_features_to_select=10),
    'SelectKBest_f': SelectKBest(score_func=f_regression, k=10),
    'PCA': PCA(n_components=0.95),
    'SPA': SPA
}

all_results = []

# Process each target variable
for target_var in y_df.columns:
    print(f"Processing target variable: {target_var}")
    y = y_df[target_var].values
    data_path = os.path.join(folder_path, final_results[target_var])
    X = pd.read_excel(data_path)
    X = X.select_dtypes(include=[np.number])  # Keep only numeric columns
    X.columns = X.columns.astype(str)  # Ensure column names are strings
    
    # Split data into Train and Test sets BEFORE applying StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)
    
    # Apply StandardScaler correctly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on Train set
    X_test_scaled = scaler.transform(X_test)  # Only transform Test set (no fit!)
    
    # Replace the original sets with the scaled versions
    X_train = X_train_scaled
    X_test = X_test_scaled
    
    """
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))  # Reshape for a single feature
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    """
 

    best_r2 = -np.inf
    best_model_info = None

    # Iterate through feature selection methods
    for method_name, selector in feature_selection_methods.items():
        print(f"Applying feature selection method: {method_name}")

        if method_name == 'No Selection':
            X_transformed = X_train
            X_test_transformed = X_test
            selected_features = 'ALL FEATURES'  # Indicate no specific selection was made
        elif method_name == 'SPA':
            selected_features_indices = selector(X_train, y_train, 10)
            selected_features = [X.columns[i] for i in selected_features_indices]  # Convert indices to feature names
            X_transformed = X_train[:, selected_features_indices]
            X_test_transformed = X_test[:, selected_features_indices]
        elif method_name == 'PCA':
            X_transformed = selector.fit_transform(X_train)
            X_test_transformed = selector.transform(X_test)
            selected_features = [f'PC{i+1}' for i in range(selector.n_components_)]  # Naming PCA components
        else:
            X_transformed = selector.fit_transform(X_train, y_train)
            X_test_transformed = selector.transform(X_test)
            if hasattr(selector, 'get_support'):
                selected_features = X.columns[selector.get_support()].tolist()  # Get selected feature names if possible
            else:
                selected_features = []  # In case the method doesn't support feature listing

        # Train and evaluate models
        for model_name, model in models.items():
            
            # Evaluate with cross-validation (on training set only)
            cv_scores = cross_val_score(model, X_transformed, y_train, cv=5, scoring='r2')
            cv_r2 = np.mean(cv_scores)
            
            # Now refit the model on the full training set
            model.fit(X_transformed, y_train)
            
            # Predictions for test set
            test_predictions = model.predict(X_test_transformed)
            test_r2 = r2_score(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

            plot_predicted_vs_actual(y_test, test_predictions, f"{target_var} – {model_name}")
            plot_residual_distribution(y_test, test_predictions, f"{target_var} – {model_name}")

            # Keep the best pipeline based on highest R2 (test set)
            
            if cv_r2 > best_r2:
                best_r2 = cv_r2
                best_model_info = {
                    'Y Variable': target_var,
                    'Feature Selection Method': method_name,
                    'Model': model_name,
                    
                    'CV R2 Score': cv_r2,
                    'Test R2 Score': test_r2,
                    'Test RMSE': test_rmse,
                    
                    'Selected Features': selected_features,
                    'y_test': y_test,
                    'Test Predictions': test_predictions
                }


           # Save results
    all_results.append(best_model_info)

# Save final results to Excel
results_df = pd.DataFrame(all_results)
results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Pipline_Per_Y\Prediction_Per_Species\Best_Pipeline_Per_Target_Per_Image.xlsx', index=False))
# results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Pipline_Per_Y\Prediction_Per_Species\Best_Pipeline_Per_Target_Per_Image_Y_Scaled.xlsx', index=False))
# results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Pipline_Per_Y\Prediction_Per_Species\Best_Pipeline_Per_Target_Per_Image_Y_Transformed.xlsx', index=False))
# results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Pipline_Per_Y\Prediction_Per_Species\Best_Pipeline_Per_Target_Per_Image_Y_Transformed_Scaled.xlsx', index=False))


