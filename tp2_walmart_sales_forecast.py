# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Loading datasets
train_file_path = "C://Users//aidan//Desktop//4AI3 Project//train.csv.zip"
features_file_path = "C://Users//aidan//Desktop//4AI3 Project//features.csv.zip"
stores_file_path = "C://Users//aidan//Desktop//4AI3 Project//stores.csv"
dataset = pd.read_csv(train_file_path, sep=',', header=0, names=['Store', 'Dept', 'Date', 'weeklySales', 'isHoliday'])
features = pd.read_csv(features_file_path, sep=',', header=0, names=['Store', 'Date', 'Temperature', 'Fuel_Price',
                                                                   'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                                                                   'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']) \
    .drop(columns=['IsHoliday'])
stores = pd.read_csv(stores_file_path, names=['Store', 'Type', 'Size'], sep=',', header=0)
dataset = dataset.merge(stores, how='left').merge(features, how='left')

# Display the newly merged dataset
print(dataset)

# Data Visualization

# Function to create scatter plots
def scatter(data, column):
    plt.figure()
    plt.scatter(data[column], data['weeklySales'])
    plt.ylabel('weeklySales')
    plt.xlabel(column)
    plt.show()

# Create plots for selected columns
scatter(dataset, 'Fuel_Price')
scatter(dataset, 'Size')
scatter(dataset, 'CPI')
scatter(dataset, 'Type')
scatter(dataset, 'isHoliday')
scatter(dataset, 'Unemployment')
scatter(dataset, 'Temperature')
scatter(dataset, 'Store')
scatter(dataset, 'Dept')

"""# Data manipulation"""

# One-hot encode the 'Type' column
dataset = pd.get_dummies(dataset, columns=["Type"])

# Fill NaN values in 4 of the MarkDown columns as 0
dataset[['MarkDown1', 'MarkDown2', 'MarkDown4', 'MarkDown5']] = dataset[['MarkDown1', 'MarkDown2', 'MarkDown4', 'MarkDown5']].fillna(0)

# Extract the month from the 'Date' column
dataset['Month'] = pd.to_datetime(dataset['Date']).dt.month

# Drop unnecessary columns
dataset = dataset.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

print(dataset)

"""# Algorithms"""

# Function to create a k-Nearest Neighbors model
def knn_model():
    knn = KNeighborsRegressor(n_neighbors=10)
    return knn

# Function to create a Support Vector Regressor model
def svm_model():
    clf = SVR(kernel='rbf', gamma='auto')
    return clf

# Function to create a Multi-layer Perceptron Regressor model
def nn_model():
    clf = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', verbose=3)
    return clf

# Function to create an Extra Trees Regressor model
def extra_trees_model():
    clf = ExtraTreesRegressor(n_estimators=100, max_features='auto', verbose=1, n_jobs=1)
    return clf

# Function to create a Random Forest Regressor model
def random_forest_model():
    clf = RandomForestRegressor(n_estimators=100, max_features='log2', verbose=1)
    return clf

# Function to make predictions using a trained model
def predict(model, test_data):
    return pd.Series(model.predict(test_data))

# Function to select a machine learning model
def select_model():
    return extra_trees_model()

# Function to train a model
def train_model(train_data, train_labels):
    model = select_model()
    model.fit(train_data, train_labels)
    return model

# Function to train a model and make predictions
def train_and_predict(train_data, train_labels, test_data):
    trained_model = train_model(train_data, train_labels)
    predictions = predict(trained_model, test_data)
    return predictions, trained_model

# Function to calculate mean absolute error
def calculate_error(true_labels, predicted_labels, weights):
    return mean_absolute_error(true_labels, predicted_labels, sample_weight=weights)

"""# K-Fold Cross Validation"""

# Split the dataset into K-folds for cross-validation
kf = KFold(n_splits=5)
splited_data = []

for name, group in dataset.groupby(["Store", "Dept"]):
    group = group.reset_index(drop=True)
    if group.shape[0] <= 5:
        fold_indices = np.array(range(5))
        np.random.shuffle(fold_indices)
        group['fold'] = fold_indices[:group.shape[0]]
    else:
        fold = 0
        for train_index, test_index in kf.split(group):
            group.loc[test_index, 'fold'] = fold
            fold += 1
    splited_data.append(group)

# Concatenate the split data
splited_data = pd.concat(splited_data).reset_index(drop=True)

print(splited_data)

best_model = None
error_cv = 0
best_error = np.iinfo(np.int32).max

# Perform K-Fold Cross Validation
for fold in range(5):
    train_set = splited_data.loc[splited_data['fold'] != fold]
    test_set = splited_data.loc[splited_data['fold'] == fold]

    train_labels = train_set['weeklySales']
    train_data = train_set.drop(columns=['weeklySales', 'fold'])

    test_labels = test_set['weeklySales']
    test_data = test_set.drop(columns=['weeklySales', 'fold'])

    print(f"Fold {fold + 1}:")

    # Train and predict
    predicted_results, current_model = train_and_predict(train_data, train_labels, test_data)

    # Adjust weights for holidays
    weights = test_data['isHoliday'].replace(True, 5).replace(False, 1)

    # Calculate mean absolute error
    error = calculate_error(test_labels, predicted_results, weights)

    print(f"MAE: {error}")

    mean_true_value = test_labels.mean()
    mape = (error / mean_true_value) * 100
    print(f"MAPE: {mape:.2f}%")

    # Update the best model if the current model has a lower error
    if error < best_error:
        print('Found best model')
        best_error = error
        best_model = current_model

# Calculate average cross-validation error
error_cv /= 5
print("Average MAE:", error_cv)
print("Best MAE:", best_error)

"""# Test part"""

# Load the test dataset
test_file_path = "C://Users//aidan//Desktop//4AI3 Project//test.csv.zip"
features_file_path_test = "C://Users//aidan//Desktop//4AI3 Project//features.csv.zip"
dataset_test = pd.read_csv(test_file_path, names=['Store', 'Dept', 'Date', 'isHoliday'], sep=',', header=0)
features_test = pd.read_csv(features_file_path_test, sep=',', header=0,
                            names=['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
                                   'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']) \
    .drop(columns=['IsHoliday'])
stores_test = pd.read_csv(stores_file_path, names=['Store', 'Type', 'Size'], sep=',', header=0)
dataset_test = dataset_test.merge(stores_test, how='left').merge(features_test, how='left')

# Preprocess the test dataset
dataset_test = pd.get_dummies(dataset_test, columns=["Type"])
dataset_test[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = \
    dataset_test[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
dataset_test = dataset_test.fillna(0)
column_date_test = dataset_test['Date']
dataset_test['Month'] = pd.to_datetime(dataset_test['Date']).dt.month
dataset_test = dataset_test.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

print(dataset_test)

# Make predictions on the test dataset using the best model
predicted_test_results = best_model.predict(dataset_test)

# Prepare the final output dataframe
dataset_test['weeklySales'] = predicted_test_results
dataset_test['Date'] = column_date_test
dataset_test['id'] = dataset_test['Store'].astype(str) + '_' + dataset_test['Dept'].astype(str) + '_' + \
                     dataset_test['Date'].astype(str)
dataset_test = dataset_test[['id', 'weeklySales']]
dataset_test = dataset_test.rename(columns={'id': 'Id', 'weeklySales': 'Weekly_Sales'})

# Save the predictions to a CSV file
output_file_path = "C://Users//aidan//Desktop//4AI3 Project//output.csv"
dataset_test.to_csv(output_file_path, index=False)