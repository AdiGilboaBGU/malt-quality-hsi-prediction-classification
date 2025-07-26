import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

##############################################################################################################################
#### COMPARE DATASET PER SPECIES FOR CLASSIFICATION
##############################################################################################################################
##############################################################################################################################

#### ALSO 2 AND 3 CLASSIFICATION

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Function to compute cross-validated accuracy
def accuracy_cv(model, X, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X, y, scoring="accuracy", cv=kf)
    return accuracy.mean()

def train_models(X_train, X_test, y_train, y_test):
    """Train different classifiers and return their accuracy and F1-score."""
    models = {
        'SVC': SVC(kernel="linear", C=1),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
#        'RandomForest': RandomForestClassifier(n_estimators=100),
        'LDA': LinearDiscriminantAnalysis(),
#        'NaiveBayes': GaussianNB()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        accuracy_cv_score = accuracy_cv(model, X_train, y_train)
        
        results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'CV Accuracy': accuracy_cv_score}

        # Plot confusion matrix for each model
        plot_confusion_matrix(y_test, predictions, f"Confusion Matrix: {name}")
    
    return results

#### Binary Case
########################################################################

datasets_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_species'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_classification\binary_data.xlsx', index_col=0)

#### Ternary Case
########################################################################

"""
# datasets_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_species'
# y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_species_classification\ternary_data.xlsx', index_col=0)
"""

# Initialize a dictionary to store the best results for each Y variable
final_results = {} 

# Process each classification target independently
for target_var in y_df.columns:
    print(f"Processing target variable: {target_var}")
    y = y_df[target_var]

  
    # Initialize results summary and best dataset tracking
    results_summary = []
    best_results = None
    best_dataset = None

    for file_name in os.listdir(datasets_path):
        if file_name.endswith('.xlsx') and not file_name.startswith('~$'):
            reflectance_df = pd.read_excel(os.path.join(datasets_path, file_name))
            X = reflectance_df.drop(columns=['Labels'])  
            X.columns = range(X.shape[1])  # Normalize column names to sequential numbers

            # Split data into Train and Test sets BEFORE applying StandardScaler
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)
            
            # Apply StandardScaler correctly
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on Train set
            X_test_scaled = scaler.transform(X_test)  # Only transform Test set (no fit!)
            
            # Replace the original sets with the scaled versions
            X_train = X_train_scaled
            X_test = X_test_scaled

            # Train and evaluate models
            results = train_models(X_train, X_test, y_train, y_test)
            results['dataset'] = file_name
            results_summary.append(results)
            
            # Track best dataset based on model performance
            if best_results is None:
                best_results = results
                best_dataset = file_name

            better_models_count = sum(
                1 for model in results if model != 'dataset' and results[model]['Accuracy'] > best_results[model]['Accuracy']
            )
            
            print(f"Processing {file_name}: better_models_count = {better_models_count}")
            
            if better_models_count >= 2:
                best_dataset = file_name
                best_results = results

    # Save the best results for this target variable
    final_results[target_var] = {
        'Best Dataset': best_dataset,
        'Best Results': best_results
    }

# Save final results to Excel
final_results_df = pd.DataFrame(final_results)
final_results_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Species_Classification_2.xlsx', index=False)

print("Results saved successfully!")



##############################################################################################################################
#### COMPARE DATASET PER SEED FOR CLASSIFICATION
##############################################################################################################################
##############################################################################################################################

#### ALSO 2 AND 3 CLASSIFICATION

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Function to compute cross-validated accuracy
def accuracy_cv(model, X, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X, y, scoring="accuracy", cv=kf)
    return accuracy.mean()

def train_models(X_train, X_test, y_train, y_test):
    """Train different classifiers and return their accuracy and F1-score."""
    models = {
        'SVC': SVC(),
        'XGB': XGBClassifier(verbosity=0),
#        'RandomForest': RandomForestClassifier(n_estimators=100),
        'LDA': LinearDiscriminantAnalysis(),
#        'NaiveBayes': GaussianNB()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        accuracy_cv_score = accuracy_cv(model, X_train, y_train)
        
        results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'CV Accuracy': accuracy_cv_score}

        # Plot confusion matrix for each model
        plot_confusion_matrix(y_test, predictions, f"Confusion Matrix: {name}")
    
    return results

#### Binary Case
########################################################################

datasets_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_classification\expanded_binary_data.xlsx', index_col=0)

#### Ternary Case
########################################################################

"""
# datasets_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_classification\expanded_ternary_data.xlsx', index_col=0)
"""

# Initialize a dictionary to store the best results for each Y variable
final_results = {} 

# Process each classification target independently
for target_var in y_df.columns:
    print(f"Processing target variable: {target_var}")
    y = y_df[target_var]

    
    # Initialize results summary and best dataset tracking
    results_summary = []
    best_results = None
    best_dataset = None

    for file_name in os.listdir(datasets_path):
        if file_name.endswith('.xlsx') and not file_name.startswith('~$'):
            X = pd.read_excel(os.path.join(datasets_path, file_name), index_col=0)
#            X = reflectance_df.drop(columns=['Labels'])  
#            X.columns = range(X.shape[1])  # Normalize column names to sequential numbers

            # Split data into Train and Test sets BEFORE applying StandardScaler
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)
            
            # Apply StandardScaler correctly
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on Train set
            X_test_scaled = scaler.transform(X_test)  # Only transform Test set (no fit!)
            
            # Replace the original sets with the scaled versions
            X_train = X_train_scaled
            X_test = X_test_scaled

            # Train and evaluate models
            results = train_models(X_train, X_test, y_train, y_test)
            results['dataset'] = file_name
            results_summary.append(results)
            
            # Track best dataset based on model performance
            if best_results is None:
                best_results = results
                best_dataset = file_name

            better_models_count = sum(
                1 for model in results if model != 'dataset' and results[model]['Accuracy'] > best_results[model]['Accuracy']
            )
            
            print(f"Processing {file_name}: better_models_count = {better_models_count}")
            
            if better_models_count >= 2:
                best_dataset = file_name
                best_results = results

    # Save the best results for this target variable
    final_results[target_var] = {
        'Best Dataset': best_dataset,
        'Best Results': best_results
    }

# Save final results to Excel
final_results_df = pd.DataFrame(final_results)
final_results_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Seed_Classification_2.xlsx', index=False)

print("Results saved successfully!")




