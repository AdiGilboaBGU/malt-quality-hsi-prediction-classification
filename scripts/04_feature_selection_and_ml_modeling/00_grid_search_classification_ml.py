import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from collections import Counter


#### Classification - Hyper - parameter Tuning With Grid Search CV With K = 5, Per Target Y, The Final Values
#### Of The Hyper - parameters Per Model Are Set By The Common Values Across All Y's
#######################################################################################################################
#######################################################################################################################

# Define paths
folder_path = os.path.join('..', '..', 'datasets', '\X_by_seed_SG_SD.xlsx')
y_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'Y_by_seed_classification\expanded_binary_data.xlsx', index_col=0))

# Load the entire dataset once
X = pd.read_excel(folder_path, index_col=0)

# Define models and hyperparameters
models = {
    'SVC': SVC(),
    'RandomForest': RandomForestClassifier(),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
    'LDA': LinearDiscriminantAnalysis(),
    'NaiveBayes': GaussianNB()
}

hyperparameters = {
    'SVC': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    'XGB': {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200]},
    'LDA': {'solver': ['svd', 'lsqr', 'eigen']},
    'NaiveBayes': {}
}

# Initialize a dictionary to collect the best parameters across all Y variables
param_aggregation = {model: [] for model in models}

# Extract group identifiers - assuming the index might represent group identification
groups = X.index.to_series().apply(lambda x: x.split('_')[1]).astype(int).values  # Adjust this line if a different column represents groups

# GroupShuffleSplit configuration
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=77)

for target_var in y_df.columns:
    y = y_df[target_var].values
    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        # Standardizing the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
#        X_test_scaled = scaler.transform(X_test)  # Transform the test set

        # Perform tuning for each model
        for model_name, model in models.items():
            if hyperparameters[model_name]:  # Ensure there are parameters to tune
                grid_search = GridSearchCV(model, hyperparameters[model_name], cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                best_params = grid_search.best_params_
                param_aggregation[model_name].append(best_params)

# Analyze the most frequent best parameters per model
final_results = []
for model_name, params_list in param_aggregation.items():
    if params_list:
        # Flatten parameter dictionaries to compare easily
        flat_params = [tuple(sorted(params.items())) for params in params_list]
        most_common_params = Counter(flat_params).most_common(1)[0][0]
        best_params = dict(most_common_params)
        final_results.append({'Model': model_name, 'Best Params': best_params})

# Save results to Excel
final_results_df = pd.DataFrame(final_results)
final_results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Optimal_Models_Results.xlsx', index=False))
print("Optimal results saved to Excel successfully!")







