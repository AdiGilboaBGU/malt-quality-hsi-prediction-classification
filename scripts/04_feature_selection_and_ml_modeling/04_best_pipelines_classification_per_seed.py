#### CLASSIFICATION WITH GROUPS WITH MAJOTIRY VOTE
############################################################################
############################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#### Best pipeline classification per seed with groups and Balanced


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Function to load data and create group identifiers from Excel
def load_excel_data_and_groups(features_file):
    """
    Load seed data from an Excel file and create group identifiers.

    Parameters:
    features_file (str): Path to the Excel file.

    Returns:
    X (numpy array): Feature matrix.
    groups (numpy array): Group identifiers based on the seed information.
    """
    # Load the data
    X = pd.read_excel(features_file, index_col=0)  # Load features with the first column as the index

    # Extract group identifiers from the index (or another column, if needed)
    groups = X.index.to_series().astype(str).apply(lambda x: x.split('_')[1] if '_' in x else x).astype(int).values

    return X, groups

# Function to compute majority vote accuracy per group
def predict_majority_vote_per_seed(model, X, groups, y_true):
    """
    Predict using the model and determine the label by majority vote per group.
    Then apply the group label to all its seeds and compute accuracy at the seed level.
    """
    # Predict class labels
    predictions = model.predict(X)

    # Aggregate predictions by group using majority vote
    df = pd.DataFrame({'group': groups, 'predictions': predictions})
    majority_votes = df.groupby('group')['predictions'].agg(lambda x: x.value_counts().idxmax()).to_dict()

    # Apply the group prediction to all its seeds
    seeds_predictions = np.array([majority_votes[g] for g in groups])

    # Compute accuracy at the seed level
    accuracy = accuracy_score(y_true, seeds_predictions)

    return accuracy

#### Binary Case
########################################################################

folder_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_classification\expanded_binary_data.xlsx', index_col=0)

# Load selected pipelines file (contains chosen feature selection and model per target variable)
selected_pipelines_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Classification_Per_Species\Best_Pipeline_Per_Target_Per_Image_2.xlsx')
selected_pipelines = selected_pipelines_df.set_index('Y Variable').to_dict(orient='index')

final_results_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Seed_Classification_2.xlsx', header=None)
final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}

# Define classification models
models = {
    'SVC': SVC(kernel="linear", C=10, gamma="scale", probability=True),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'LDA': LinearDiscriminantAnalysis(),
    'NaiveBayes': GaussianNB()
}

# Define feature selection techniques for classification
feature_selection_methods = {
    'No Selection': None,  
#    'RFE': RFE(estimator=SVC(kernel="linear"), n_features_to_select=10),
    'SelectKBest_f': SelectKBest(score_func=f_classif, k=10),
    'PCA': PCA(n_components=0.95)
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
    X_train = scaler.fit_transform(X_train)  
    X_test = scaler.transform(X_test)  

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
            
            
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
    
    # Train and evaluate the selected model
    model = models[selected_model]
    model.fit(X_transformed, y_train)
    
    # Predictions for training set
    train_predictions = model.predict(X_transformed)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions, average='weighted')
    train_majority_vote_accuracy = predict_majority_vote_per_seed(model, X_transformed, groups_train, y_train)
    
    # Predictions for test set
    test_predictions = model.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    test_majority_vote_accuracy = predict_majority_vote_per_seed(model, X_test_transformed, groups_test, y_test)
    
    # Compute AUC for the entire dataset
    train_auc = roc_auc_score(y_train, model.predict_proba(X_transformed)[:, 1]) if hasattr(model, "predict_proba") else None
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1]) if hasattr(model, "predict_proba") else None
    
    # Compute F1 Score, Recall, and Precision for each class
    train_f1_per_class = f1_score(y_train, train_predictions, average=None)
    test_f1_per_class = f1_score(y_test, test_predictions, average=None)
    
    train_recall_per_class = recall_score(y_train, train_predictions, average=None)
    test_recall_per_class = recall_score(y_test, test_predictions, average=None)
    
    train_precision_per_class = precision_score(y_train, train_predictions, average=None)
    test_precision_per_class = precision_score(y_test, test_predictions, average=None)
    
    # Save combined training and testing results
    best_model_info = {
        'Y Variable': target_var,
        'Feature Selection Method': selected_method,
        'Model': selected_model,
        
        # Accuracy
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        
        # Majority Vote Accuracy
        'Train Majority Vote Accuracy': train_majority_vote_accuracy,
        'Test Majority Vote Accuracy': test_majority_vote_accuracy,

        # F1 Score
        'Train F1 Score': train_f1,
        'Test F1 Score': test_f1,      
       
        # AUC
        'Train AUC': train_auc,
        'Test AUC': test_auc,
        
        # F1 Score per class
        'Train F1 Score Class 0': train_f1_per_class[0],
        'Train F1 Score Class 1': train_f1_per_class[1],
        'Test F1 Score Class 0': test_f1_per_class[0],
        'Test F1 Score Class 1': test_f1_per_class[1],
        
        # Recall per class
        'Train Recall Class 0': train_recall_per_class[0],
        'Train Recall Class 1': train_recall_per_class[1],
        'Test Recall Class 0': test_recall_per_class[0],
        'Test Recall Class 1': test_recall_per_class[1],
        
        # Precision per class
        'Train Precision Class 0': train_precision_per_class[0],
        'Train Precision Class 1': train_precision_per_class[1],
        'Test Precision Class 0': test_precision_per_class[0],
        'Test Precision Class 1': test_precision_per_class[1],
        
        # Selected Features
        'Selected Features': selected_features
    }
    
    all_results.append(best_model_info)

    
# Save final results to Excel
results_df = pd.DataFrame(all_results)
results_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Classification_Per_Seed\Best_Results_With_Groups_Per_Seed.xlsx', index=False)

print("Results saved successfully!")

    

#### CLASSIFICATION WITH GROUPS WITH MAJOTIRY VOTE
############################################################################
############################################################################

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Function to load data and create group identifiers from Excel
def load_excel_data_and_groups(features_file):
    X = pd.read_excel(features_file, index_col=0)
    groups = X.index.to_series().astype(str).apply(lambda x: x.split('_')[1] if '_' in x else x).astype(int).values
    return X, groups

# Function to compute majority vote accuracy per group
def predict_majority_vote_per_seed(model, X, groups, y_true):
    predictions = model.predict(X)
    df = pd.DataFrame({'group': groups, 'predictions': predictions})
    majority_votes = df.groupby('group')['predictions'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    seeds_predictions = np.array([majority_votes[g] for g in groups])
    accuracy = accuracy_score(y_true, seeds_predictions)
    return accuracy

# Function to split groups into Train (80%) and Test (20%)
def split_groups(groups_list):
    train_groups, test_groups = train_test_split(groups_list, test_size=0.2, random_state=77)  
    return train_groups, test_groups

#### Binary Case
########################################################################

folder_path = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed'
y_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_classification\expanded_binary_data.xlsx', index_col=0)

# Load selected pipelines file (contains chosen feature selection and model per target variable)
selected_pipelines_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Classification_Per_Species\Best_Pipeline_Per_Target_Per_Image_2.xlsx')
selected_pipelines = selected_pipelines_df.set_index('Y Variable').to_dict(orient='index')

final_results_df = pd.read_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Seed_Classification_2.xlsx', header=None)
final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}

# Define classification models
models = {
    'SVC': SVC(kernel="linear", C=10, gamma="scale", probability=True),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'LDA': LinearDiscriminantAnalysis(),
    'NaiveBayes': GaussianNB()
}

# Define feature selection techniques for classification
feature_selection_methods = {
    'No Selection': None,  
#    'RFE': RFE(estimator=SVC(kernel="linear"), n_features_to_select=10),
    'SelectKBest_f': SelectKBest(score_func=f_classif, k=10),
    'PCA': PCA(n_components=0.95)
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

    # === Step 1: Identify unique groups per class ===
    group_class_map = {g: y[groups == g][0] for g in np.unique(groups)}
    groups_0 = [g for g in group_class_map if group_class_map[g] == 0]
    groups_1 = [g for g in group_class_map if group_class_map[g] == 1]

    # === Step 2: Split groups while keeping class balance ===
    train_groups_0, test_groups_0 = split_groups(groups_0)
    train_groups_1, test_groups_1 = split_groups(groups_1)

    # Merge groups of both classes
    train_groups = np.array(train_groups_0 + train_groups_1)
    test_groups = np.array(test_groups_0 + test_groups_1)

    # === Step 3: Create masks for dataset filtering ===
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups_train = groups[train_mask]
    groups_test = groups[test_mask]

    # === Step 4: Standardize the data ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
    
    # Train and evaluate the selected model
    model = models[selected_model]
    model.fit(X_transformed, y_train)
    
    # Predictions for training set
    train_predictions = model.predict(X_transformed)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions, average='weighted')
    train_majority_vote_accuracy = predict_majority_vote_per_seed(model, X_transformed, groups_train, y_train)
    
    # Predictions for test set
    test_predictions = model.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    test_majority_vote_accuracy = predict_majority_vote_per_seed(model, X_test_transformed, groups_test, y_test)
    
    # Compute AUC for the entire dataset
    train_auc = roc_auc_score(y_train, model.predict_proba(X_transformed)[:, 1]) if hasattr(model, "predict_proba") else None
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1]) if hasattr(model, "predict_proba") else None
    
    # Compute F1 Score, Recall, and Precision for each class
    train_f1_per_class = f1_score(y_train, train_predictions, average=None)
    test_f1_per_class = f1_score(y_test, test_predictions, average=None)
    
    train_recall_per_class = recall_score(y_train, train_predictions, average=None)
    test_recall_per_class = recall_score(y_test, test_predictions, average=None)
    
    train_precision_per_class = precision_score(y_train, train_predictions, average=None)
    test_precision_per_class = precision_score(y_test, test_predictions, average=None)
    
    # Save combined training and testing results
    best_model_info = {
        'Y Variable': target_var,
        'Feature Selection Method': selected_method,
        'Model': selected_model,
        
        # Accuracy
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,

        # Majority Vote Accuracy
        'Train Majority Vote Accuracy': train_majority_vote_accuracy,
        'Test Majority Vote Accuracy': test_majority_vote_accuracy,
        
        # F1 Score
        'Train F1 Score': train_f1,
        'Test F1 Score': test_f1,
            
        # AUC
        'Train AUC': train_auc,
        'Test AUC': test_auc,
        
        # F1 Score per class
        'Train F1 Score Class 0': train_f1_per_class[0],
        'Train F1 Score Class 1': train_f1_per_class[1],
        'Test F1 Score Class 0': test_f1_per_class[0],
        'Test F1 Score Class 1': test_f1_per_class[1],
        
        # Recall per class
        'Train Recall Class 0': train_recall_per_class[0],
        'Train Recall Class 1': train_recall_per_class[1],
        'Test Recall Class 0': test_recall_per_class[0],
        'Test Recall Class 1': test_recall_per_class[1],
        
        # Precision per class
        'Train Precision Class 0': train_precision_per_class[0],
        'Train Precision Class 1': train_precision_per_class[1],
        'Test Precision Class 0': test_precision_per_class[0],
        'Test Precision Class 1': test_precision_per_class[1],
        
        # Selected Features
        'Selected Features': selected_features
    }
    
    all_results.append(best_model_info)

# Save results to Excel
results_df = pd.DataFrame(all_results)
results_df.to_excel(r'G:\My Drive\Thesis\Temp_Work\excel_files_final\ML_results\Best_Pipline_Per_Y\Classification_Per_Seed\Best_Results_With_Groups_Balanced_Per_Seed.xlsx', index=False)

print("Results saved successfully!")

