import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras import backend as K
from collections import Counter
from collections import defaultdict


#### Classification - Hyper - parameter Tuning With Grid Search CV With K = 5, Per Target Y, The Final Values
#### Of The Hyper - parameters Per Model Are Set By The Common Values Across All Y's
#######################################################################################################################
#######################################################################################################################


# Load the data
features_file = os.path.join('..', '..', 'datasets', 'X_by_seed\X_by_seed_SG_SD.xlsx')
labels_file = os.path.join('..', '..', 'datasets', 'Y_by_seed_classification\expanded_binary_data.xlsx')

X_df = pd.read_excel(features_file, index_col=0)
y_df = pd.read_excel(labels_file, index_col=0)

def create_model(learning_rate=0.001, optimizer_name='Adam', dropout_rate=0.3, alpha=0.3, num_filters=[16, 32], kernel_size=3, strides=1, padding_type='same', regularization=0.01, activation='relu'):
    model = Sequential()
    
    # Function to select activation
    def get_activation(x, act_type, alpha=None):
        if act_type == 'relu':
            return relu(x)
        elif act_type == 'tanh':
            return tanh(x)
        elif act_type == 'leaky_relu':
            if alpha is None:
                alpha = 0.3  # Default alpha
            return LeakyReLU(alpha)(x)
        else:
            return x  # No activation, linear

    # Convolution Block 1
    model.add(Conv1D(num_filters[0], kernel_size=kernel_size, strides=strides, padding=padding_type,
                     kernel_regularizer=l2(regularization), input_shape=(X_df.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Activation(lambda x: get_activation(x, activation, alpha)))
    model.add(Dropout(dropout_rate))
    
    # Convolution Block 2
    model.add(Conv1D(num_filters[1], kernel_size=kernel_size, strides=strides, padding=padding_type, kernel_regularizer=l2(regularization)))
    model.add(BatchNormalization())
    model.add(Activation(lambda x: get_activation(x, activation, alpha)))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(lambda x: get_activation(x, activation, alpha)))
    model.add(Dropout(0.3))
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation(lambda x: get_activation(x, activation, alpha)))
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate) if optimizer_name == 'Adam' else \
                SGD(learning_rate=learning_rate) if optimizer_name == 'SGD' else \
                RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

# Process each parameter
param_aggregation = []

for column in y_df.columns:
    print(f"Processing parameter: {column}")
    y = y_df[column].values
    groups = X_df.index.to_series().apply(lambda x: x.split('_')[1]).astype(int).values
    
    # Perform GroupShuffleSplit to ensure group separation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=77)
    for train_idx, _ in gss.split(X_df, y, groups):
        X_train, y_train = X_df.iloc[train_idx], y[train_idx]
        groups_train = groups[train_idx]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define the grid search parameters
    param_grid = {
        'batch_size': [16, 32],
        'epochs': [50, 100],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'alpha': [0.1, 0.2, 0.3],
        'num_filters': [(16, 32), (32, 64)],
        'kernel_size': [3, 5, 7],
        'strides': [1, 2],
        'padding_type': ['same', 'valid'],
        'regularization': [0.001, 0.01],
        'optimizer_name': ['Adam', 'SGD', 'RMSprop'],
        'activation': ['relu', 'tanh', 'leaky_relu']
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
    grid_result = grid.fit(X_train_scaled, y_train)

    # Aggregate parameter results for further analysis
    best_params = grid_result.best_params_
    param_aggregation.append(grid_result.best_params_)


# Analyze the most frequent best parameters across all targets
flat_params = [tuple(sorted(params.items())) for params in param_aggregation]
most_common_params = Counter(flat_params).most_common(1)[0][0]
best_params = dict(most_common_params)

# Save to Excel
final_results_df = pd.DataFrame([{'Model': 'CNN', 'Best Params': best_params}])
final_results_df.to_excel(os.path.join('..', '..', 'datasets', 'ML_results\Optimal_CNN_Results.xlsx', index=False))
print("Optimal CNN parameters saved to Excel successfully!")







