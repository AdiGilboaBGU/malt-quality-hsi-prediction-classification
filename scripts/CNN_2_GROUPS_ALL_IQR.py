import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

#### GROUPS 2 ALL SEED 77 IQR

# Function to plot training and validation loss and accuracy
def plot_history(history, param_name):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {param_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_plots_groups_2_classes\{param_name}_training_validation_loss.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy for {param_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_plots_groups_2_classes\{param_name}_training_validation_accuracy.png')
    plt.close()


# Function to plot density and ROC curves
def plot_density_and_roc(X_test, y_test, model, param_name):
    y_pred_proba = model.predict(X_test).flatten()
    y_test = y_test.flatten()  # True labels of the test set
    results_df = pd.DataFrame({'True Label': y_test, 'Predicted Probability': y_pred_proba})
    
    # Density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=results_df[results_df['True Label'] == 0], x='Predicted Probability', fill=True, color='green', alpha=0.6, label='Class 0')
    sns.kdeplot(data=results_df[results_df['True Label'] == 1], x='Predicted Probability', fill=True, color='red', alpha=0.6, label='Class 1')
    plt.title(f'Density Plot of Predicted Probabilities for {param_name}')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_plots_groups_2_classes\{param_name}_density_plot.png')
    plt.close()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {param_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_plots_groups_2_classes\{param_name}_roc_curve_plot.png')
    plt.close()


# Function to remove outliers based on IQR in multiple features
def remove_outliers_iqr(X, y, groups):
    outlier_indices = []

    # Iterate over each feature in X
    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        column_outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)].index
        outlier_indices.extend(column_outliers)

    # Count occurrences of outliers in multiple features
    outlier_count = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_count.items() if v > 1]  # Keep only records that are outliers in more than one feature

    print(f"Total outliers detected in multiple features: {len(multiple_outliers)}")

    # Remove identified outliers from X, y, and groups
    X_cleaned = X.drop(multiple_outliers)

    # Ensure only indices present in y and groups are removed
    multiple_outliers = [idx for idx in multiple_outliers if idx in y.index]
    y_cleaned = y.drop(multiple_outliers)

    # Ensure groups is also filtered
    groups_cleaned = groups[np.isin(X.index, y_cleaned.index)]

    print(f"Dataset size after outlier removal: {X_cleaned.shape[0]} samples")

    return X_cleaned, y_cleaned, groups_cleaned



# Evaluate and generate metrics for the model
def evaluate_model(model, X, y, dataset_name, param_name):
    # Predict on data
    predictions = model.predict(X)

    # Convert predictions to binary labels (0 or 1)
    y_pred_classes = (predictions > 0.5).astype(int).flatten()

    # True labels (already binary)
    y_classes = y.astype(int).flatten()

    # Generate and print the classification report
    report = classification_report(y_classes, y_pred_classes, target_names=['Class 0', 'Class 1'], output_dict=True)
    print(f"{param_name} {dataset_name} Classification Report:")
    print(report)

    # Generate and display confusion matrix
    conf_matrix = confusion_matrix(y_classes, y_pred_classes)
    print(f"{param_name} {dataset_name} Confusion Matrix:")
    print(conf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{param_name} {dataset_name} Confusion Matrix")
    plt.show()

    # Calculate AUC score
    auc_score = roc_auc_score(y_classes, predictions.flatten())  # Correct AUC calculation with flattened predictions

    # Evaluate the model
    results = model.evaluate(X, y, verbose=0)  # results now contain loss and additional metrics

    # Prepare metrics dictionary
    metrics = {
        'Loss': results[0],
        'Accuracy': report['accuracy'],
        'AUC': auc_score,
        'F1 Score Class 0': report['Class 0']['f1-score'],
        'F1 Score Class 1': report['Class 1']['f1-score'],
        'Recall Class 0': report['Class 0']['recall'],
        'Recall Class 1': report['Class 1']['recall'],
        'Precision Class 0': report['Class 0']['precision'],
        'Precision Class 1': report['Class 1']['precision']
    }
    
    # Print the results
    print(f"{param_name} {dataset_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics, conf_matrix



from sklearn.metrics import accuracy_score

def predict_majority_vote_per_seed(model, X, groups, y_true, dataset_name="Dataset"):
    """
    Predict using the model and determine the label by majority vote per group.
    Then apply the group label to all its seeds and compute accuracy at the seed level.
    """
    # Predict probabilities for each sample
    predictions = model.predict(X)
    predictions_classes = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to class labels (0 or 1)

    # Aggregate predictions by group using majority vote
    df = pd.DataFrame({'group': groups, 'predictions': predictions_classes})
    majority_votes = df.groupby('group')['predictions'].agg(lambda x: x.value_counts().idxmax()).to_dict()

    # Apply the group prediction to all its seeds
    seeds_predictions = np.array([majority_votes[g] for g in groups])

    # Compute accuracy at the seed level (all samples)
    accuracy = accuracy_score(y_true, seeds_predictions)

    return accuracy



def save_results_all_in_one(train_metrics, train_confusion_mtx, y_train, 
                            val_metrics, val_confusion_mtx, y_val, 
                            test_metrics, test_confusion_mtx, y_test, 
                            model, X_train, groups_train, X_val, groups_val, 
                            X_test, groups_test, param_name):
    """
    Save Train, Validation, and Test results into a single Excel file, 
    with additional evaluation based on majority vote per seed.
    """
    # Calculate seed level accuracy using majority vote
    train_seed_accuracy = predict_majority_vote_per_seed(model, X_train, groups_train, y_train, "Train")
    val_seed_accuracy = predict_majority_vote_per_seed(model, X_val, groups_val, y_val, "Validation")
    test_seed_accuracy = predict_majority_vote_per_seed(model, X_test, groups_test, y_test, "Test")

    # Add seed level accuracy to the metrics dictionary
    seed_metrics = {
        "Train": {"Seed Accuracy": train_seed_accuracy},
        "Validation": {"Seed Accuracy": val_seed_accuracy},
        "Test": {"Seed Accuracy": test_seed_accuracy}
    }

    # Path to save the Excel file
    file_path = fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_2_classes_groups_seed_77_IQR\results_{param_name}.xlsx'

    # Prepare results for each dataset
    results_data = {
        "Train_Metrics": pd.DataFrame([train_metrics]),
        "Validation_Metrics": pd.DataFrame([val_metrics]),
        "Test_Metrics": pd.DataFrame([test_metrics]),
        "Train_Confusion_Matrix": pd.DataFrame(train_confusion_mtx, index=['Class 0', 'Class 1'], columns=['Class 0', 'Class 1']),
        "Validation_Confusion_Matrix": pd.DataFrame(val_confusion_mtx, index=['Class 0', 'Class 1'], columns=['Class 0', 'Class 1']),
        "Test_Confusion_Matrix": pd.DataFrame(test_confusion_mtx, index=['Class 0', 'Class 1'], columns=['Class 0', 'Class 1']),
        "Class Distribution": pd.DataFrame({
            'Training': np.bincount(y_train),
            'Validation': np.bincount(y_val),
            'Test': np.bincount(y_test)
        }),
        "Seed_Level_Accuracy": pd.DataFrame(seed_metrics)
    }

    # Save to Excel with different sheets for each result
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        for sheet_name, df in results_data.items():
            df.to_excel(writer, sheet_name=sheet_name)

    print(f"Results saved in one file: {file_path}")


# Load processed data
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
    groups = X.index.to_series().apply(lambda x: x.split('_')[1]).astype(int).values  # Assuming group is encoded in index

    # Convert features to a numpy array
    
    return X, groups

# Function to create and compile the model
def create_model(input_shape):

    model = Sequential()
    
    # Convolution Block 1
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', 
                     kernel_regularizer=l2(0.01), input_shape=(X.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.35))
    
    # Convolution Block 2
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.35))
    
    # Flatten Layer
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    # Fully Connected Layers
    
    model.add(Dense(128, activation=None))  # Third dense layer
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation=None))  # Third dense layer
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))

    
    # Output Layer 
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.00005, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    
    return model


labels_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_classification\expanded_binary_data.xlsx'
y_labels_df = pd.read_excel(labels_file, index_col=0)
features_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed\X_by_seed_SG_SD.xlsx'


for column in y_labels_df.columns:
    print(f"Processing parameter: {column}")
    y = pd.Series(np.array(y_labels_df[column]), index=y_labels_df.index)  # Convert y to pandas Series
    X, groups = load_excel_data_and_groups(features_file)
    
    # Remove outliers using IQR
    X, y, groups = remove_outliers_iqr(X, y, groups)
    X = X.to_numpy()
    y = y.to_numpy()
 
    # Prepare GroupShuffleSplit for dividing the data while keeping groups together
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=77)
    
    # Splitting the data into training and testing sets ensuring no group overlap
    for train_idx, test_idx in gss.split(X, y, groups):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        groups_train_full, groups_test = groups[train_idx], groups[test_idx]
    
    # Further split the training data into training and validation sets while keeping groups unique
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=77)
    for train_idx, val_idx in gss_val.split(X_train_full, y_train_full, groups_train_full):
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
        groups_train = groups_train_full[train_idx]
        groups_val = groups_train_full[val_idx]
    
    # Verify group uniqueness
    print("Groups in Training Set:", np.unique(groups_train))
    print("Groups in Validation Set:", np.unique(groups_val))
    print("Groups in Test Set:", np.unique(groups[test_idx]))
    
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X.shape[1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X.shape[1])).reshape(X_test.shape)
    
    print(f"Dataset size after outlier removal: {X.shape[0]} samples")
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    
    common_groups_train_val = set(groups_train).intersection(set(groups_val))
    common_groups_train_test = set(groups_train).intersection(set(groups[test_idx]))
    common_groups_val_test = set(groups_val).intersection(set(groups[test_idx]))
    
    print(f"Groups shared between Train and Validation: {common_groups_train_val}")
    print(f"Groups shared between Train and Test: {common_groups_train_test}")
    print(f"Groups shared between Validation and Test: {common_groups_val_test}")

    print("Original X shape:", X.shape[1])
    print("Train X shape:", X_train.shape[1])
    
    # Build and train model
    model = create_model(X.shape[1])
    # Print summary for the input and layers
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    # Evaluate the model and save metrics
    train_metrics, train_confusion_mtx = evaluate_model(model, X_train, y_train, "Train", column)
    val_metrics, val_confusion_mtx = evaluate_model(model, X_val, y_val, "Validation", column)
    test_metrics, test_confusion_mtx = evaluate_model(model, X_test, y_test, "Test", column)
    

    # Save results to Excel
    
    # Save results to Excel
    save_results_all_in_one(
        train_metrics, train_confusion_mtx, y_train,
        val_metrics, val_confusion_mtx, y_val,
        test_metrics, test_confusion_mtx, y_test,
        model, X_train, groups_train, X_val, groups_val, X_test, groups_test, column
    )

    # Plot training and validation loss and accuracy
    plot_history(history, column)

    # Plot density and ROC curve
    plot_density_and_roc(X_test, y_test, model, column)    
