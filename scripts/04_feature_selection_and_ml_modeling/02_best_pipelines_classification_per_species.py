import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


#### Binary Case
########################################################################

folder_path = os.path.join('..', '..', 'datasets', 'X_by_species')
y_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'Y_by_species_classification\binary_data.xlsx', index_col=0))

final_results_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Species_Classification_2.xlsx', header=None))
final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}

#### Ternary Case
########################################################################

"""
# folder_path = os.path.join('..', '..', 'datasets', 'X_by_species')
# y_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'Y_by_species_classification\ternary_data.xlsx', index_col=0))

# final_results_df = pd.read_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Dataset_Per_Y\Best_Dataset_Per_Species_Classification_3.xlsx', header=None))
# final_results = {final_results_df.iloc[0, col]: final_results_df.iloc[1, col] for col in final_results_df.columns}
"""
########################################################################


# Define classification models
models = {
    'SVC': SVC(kernel="linear", C=1, probability=True),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'LDA': LinearDiscriminantAnalysis(),
    'NaiveBayes': GaussianNB()
}

# Define feature selection techniques for classification
feature_selection_methods = {
    'No Selection': None,  # Option without feature selection
#    'RFE': RFE(estimator=SVC(kernel="linear"), n_features_to_select=10),
    'SelectKBest_f': SelectKBest(score_func=f_classif, k=10),
    'PCA': PCA(n_components=0.95)
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
    
    # Split data into Train and Test sets BEFORE applying StandardScaler with Stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=82)
    
    # Apply StandardScaler correctly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on Train set
    X_test_scaled = scaler.transform(X_test)  # Only transform Test set (no fit!)
    
    # Replace the original sets with the scaled versions
    X_train = X_train_scaled
    X_test = X_test_scaled

    best_accuracy = -np.inf
    best_model_info = None

    # Iterate through feature selection methods
    for method_name, selector in feature_selection_methods.items():
        print(f"Applying feature selection method: {method_name}")

        if method_name == 'No Selection':
            X_transformed = X_train
            X_test_transformed = X_test
            selected_features = 'ALL FEATURES'  # Indicate no specific selection was made
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

        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
        
        # Train and evaluate models
        for model_name, model in models.items():
            cv_scores = cross_val_score(model, X_transformed, y_train, cv=5, scoring='accuracy')
            avg_cv_accuracy = np.mean(cv_scores)

            model.fit(X_transformed, y_train)
        
            # Predictions for training set
            train_predictions = model.predict(X_transformed)
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_f1 = f1_score(y_train, train_predictions, average='weighted')
            
            train_f1_per_class = f1_score(y_train, train_predictions, average=None)
            train_recall_per_class = recall_score(y_train, train_predictions, average=None)
            train_precision_per_class = precision_score(y_train, train_predictions, average=None)
            
            train_auc = roc_auc_score(y_train, model.predict_proba(X_transformed)[:, 1]) if hasattr(model, "predict_proba") else None
        
            # Predictions for test set
            test_predictions = model.predict(X_test_transformed)
            test_accuracy = accuracy_score(y_test, test_predictions)
            test_f1 = f1_score(y_test, test_predictions, average='weighted')
        
            test_f1_per_class = f1_score(y_test, test_predictions, average=None)
            test_recall_per_class = recall_score(y_test, test_predictions, average=None)
            test_precision_per_class = precision_score(y_test, test_predictions, average=None)
        
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1]) if hasattr(model, "predict_proba") else None
        
            # Keep the best pipeline based on highest test accuracy
               
            if avg_cv_accuracy > best_accuracy:
                best_accuracy = avg_cv_accuracy  
                
                best_model_info = {
                    'Y Variable': target_var,
                    'Feature Selection Method': method_name,
                    'Model': model_name,
        
                    # Accuracy
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
        
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



    # Save results
    all_results.append(best_model_info)

# Save final results to Excel
results_df = pd.DataFrame(all_results)
results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Best_Pipline_Per_Y\Classification_Per_Species\Best_Pipeline_Per_Target_Per_Image_2.xlsx', index=False))

print("Results saved successfully!")

