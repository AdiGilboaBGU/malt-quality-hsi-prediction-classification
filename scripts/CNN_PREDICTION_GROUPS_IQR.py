import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from collections import Counter

#### CNN PREDICTION GROUPS IQR 

# Function to remove outliers based on IQR in multiple features
def remove_outliers_iqr(X, y, groups):
    outlier_indices = []
    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        column_outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)].index
        outlier_indices.extend(column_outliers)
    
    outlier_count = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_count.items() if v > 1]  # Remove records that are outliers in multiple features
    print(f"Total outliers detected in multiple features: {len(multiple_outliers)}")
    
    X_cleaned = X.drop(multiple_outliers)
    y_cleaned = y.drop(multiple_outliers)
    groups_cleaned = groups[np.isin(X.index, y_cleaned.index)]
    
    print(f"Dataset size after outlier removal: {X_cleaned.shape[0]} samples")
    return X_cleaned, y_cleaned, groups_cleaned

# Function to plot training and validation loss
def plot_history(history, param_name):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {param_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_cnn_regression_groups_IQR\{param_name}_training_validation_loss.png')
    plt.close()

def predict_mean_per_group(model, X, groups, y_true):
    predictions = model.predict(X).flatten()
    df = pd.DataFrame({'Group': groups, 'Predictions': predictions})
    group_means = df.groupby('Group')['Predictions'].mean().to_dict()
    group_predictions = np.array([group_means[g] for g in groups])
    group_r2 = r2_score(y_true, group_predictions)
    return group_r2, group_predictions

# Function to evaluate model
def evaluate_model(model, X, y, groups, dataset_name, param_name):
    predictions = model.predict(X).flatten()
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    group_r2, _ = predict_mean_per_group(model, X, groups, y)
    
    metrics = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R^2': r2,
        'Group R^2': group_r2
    }
    print(f"{param_name} {dataset_name} - MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, R^2: {r2:.4f}, Group R^2: {group_r2:.4f}")
    return metrics

# Function to save results
def save_results_all_in_one(metrics, param_name):
    file_path = fr'G:\My Drive\Thesis\Temp_Work\excel_files_final\CNN_results\results_cnn_regression_groups_IQR\results_prediction_{param_name}.xlsx'
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        for dataset_name, data_metrics in metrics.items():
            df = pd.DataFrame([data_metrics])
            df.to_excel(writer, sheet_name=dataset_name)
    print(f"Results saved in one file: {file_path}")

# Function to load data and groups
def load_data_and_groups(features_file):
    X = pd.read_excel(features_file, index_col=0)
    groups = X.index.to_series().apply(lambda x: x.split('_')[1]).astype(int).values
    return X, groups



# Function to create model
def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01), input_shape=(input_shape, 1)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Conv1D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Flatten(),
        Dropout(0.5),
        BatchNormalization(),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.000125), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

"""
# Function to create model
def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01), input_shape=(input_shape, 1)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Flatten(),
        Dropout(0.5),
        BatchNormalization(),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.000125), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model
"""

features_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\X_by_seed\X_by_seed_SG_SD.xlsx'
labels_file = r'G:\My Drive\Thesis\Temp_Work\excel_files_final\Y_by_seed_prediction\Y_by_seed_prediction.xlsx'
y_labels_df = pd.read_excel(labels_file, index_col=0)

for column in y_labels_df.columns:
    print(f"Processing parameter: {column}")
    y = pd.Series(np.array(y_labels_df[column]), index=y_labels_df.index)
    X, groups = load_data_and_groups(features_file)
    
    # Remove outliers
    X, y, groups = remove_outliers_iqr(X, y, groups)
    X = X.to_numpy()
    y = y.to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=77)
    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]  # Ensure groups are split correctly
   
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=77)
    for train_idx, val_idx in gss_val.split(X_train, y_train, groups[train_idx]):
        X_train, X_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = y_train[train_idx], y_train[val_idx]
        groups_train, groups_val = groups_train[train_idx], groups_train[val_idx]  # Update group splits for validation

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = create_model(X_train.shape[1])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)
    
    train_metrics = evaluate_model(model, X_train, y_train, groups_train, "Train", column)
    val_metrics = evaluate_model(model, X_val, y_val, groups_val, "Validation", column)
    test_metrics = evaluate_model(model, X_test, y_test, groups_test, "Test", column)
    
    all_metrics = {'Train': train_metrics, 'Validation': val_metrics, 'Test': test_metrics}
    save_results_all_in_one(all_metrics, column)
    plot_history(history, column)
