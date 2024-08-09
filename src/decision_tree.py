import random

import numpy as np
import pandas as pd
import scipy
import scipy.special
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score


random.seed(42069)


# contants
# define columns
static_label_columns = ['weather', 'surface', 'track_condition']
static_numeric_columns = ['distance', 'run_up_distance']
racer_label_columns = ['meds', 'equip', 'sex']
date_columns = ['race_date']
racer_numerical_columns = ['jockey_key', 'trainer_key', 'weight', 'post_position', 'dollar_odds', 'age']
target_column = 'official_finish'


def get_df():
    df = pd.read_csv('../data/output_results_new.csv', dtype=str)
    df['num_of_racers'] = df['num_of_racers'].astype(int)
    return df

def partition_by_size(df, size):
    return df[df['num_of_racers'] == size].copy()



def single_column_to_label(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    return df

def vectorize_date(df, column):
    df[column] = pd.to_datetime(df[column]).dt.month
    return df

def convert_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def set_display_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

def process_label_columns(df, columns):
    for column in columns:
        df = single_column_to_label(df, column)
    return df

def process_date_columns(df, columns):
    for column in columns:
        df = vectorize_date(df, column)
    return df

def process_numeric_columns(df, columns):
    df = convert_to_numeric(df, columns)
    return df

def build_racer_columns(label_columns, numerical_columns, size):
    grouped_columns = []
    for i in range(size):
        racer_columns = []
        for col in label_columns:
            racer_columns.append(f'racer_{i}_{col}')
        for col in numerical_columns:
            racer_columns.append(f'racer_{i}_{col}')
        grouped_columns.append(racer_columns)
    return grouped_columns

def create_final_matrix(df, static_columns, grouped_columns):
    final_matrix = []
    for _, row in df.iterrows():
        row_vector = [row[col] for col in static_columns]
        for group in grouped_columns:
            for column in group:
                row_vector.append(row[column])
        final_matrix.append(row_vector)
    return np.array(final_matrix)


def combine_columns_into_rows(df, static_numeric_columns, static_label_columns, grouped_racer_columns):
    rows = []
    for index, row in df.iterrows():
        for group in grouped_racer_columns:
            racer_row = []
            for col in group:
                racer_row.append(row[col])
            for static_col in static_numeric_columns + static_label_columns:
                racer_row.append(row[static_col])
            rows.append(racer_row)
    return rows

def make_permutations(data_points, targets, k):
    augmented_data_points = []
    augmented_targets = []
    perm_indices = []
    for matrix, target in zip(data_points, targets):
        augmented_data_points.append(matrix)
        augmented_targets.append(target)
        num_rows = matrix.shape[0]
        unique_permutations = set()
        default_perm = tuple(range(num_rows))
        unique_permutations.add(default_perm)
        perm_indices.append(default_perm)
        while len(unique_permutations) < k:
            perm = tuple(random.sample(range(num_rows), num_rows))
            if perm not in unique_permutations:
                unique_permutations.add(perm)
                permuted_matrix = matrix[list(perm), :]
                permuted_target = [target[i] for i in perm]
                augmented_data_points.append(permuted_matrix)
                augmented_targets.append(permuted_target)
                perm_indices.append(perm)
    return augmented_data_points, augmented_targets, perm_indices



def vectorize_result_df(df, size):

    grouped_racer_columns = build_racer_columns(racer_label_columns, racer_numerical_columns, size)

    # Process static columns
    df = process_label_columns(df, static_label_columns)
    df = process_numeric_columns(df, static_numeric_columns)
    df = process_date_columns(df, date_columns)

    # Process racer-specific columns
    for group in grouped_racer_columns:
        df = process_label_columns(df, [col for col in group if any(label in col for label in racer_label_columns)])
        df = process_numeric_columns(df, [col for col in group if
                                          any(numeric in col for numeric in racer_numerical_columns)])
    df = process_numeric_columns(df, [f"racer_{i}_{target_column}" for i in range(size)])


    def combine_columns_into_matrices(df, grouped_racer_columns, target_column, size):
        data_points = []
        targets = []
        static_labels = None
        for index, row in df.iterrows():
            racer_matrix = []
            target_row = []
            for i in range(size):
                racer_row = []
                for col in grouped_racer_columns[i]:
                    racer_row.append(row[col])
                racer_matrix.append(racer_row)
                target_row.append(row[f"racer_{i}_{target_column}"])
            data_points.append(np.array(racer_matrix))
            targets.append(np.array(target_row))
            if static_labels is None:
                static_labels = [row[static_col] for static_col in static_numeric_columns + static_label_columns]
        return data_points, targets, static_labels

    # Create final list of matrices, targets, and static labels
    data_points, targets, static_labels = combine_columns_into_matrices(df, grouped_racer_columns, target_column, size)
    return np.array(data_points), np.array(targets), static_labels
def k_func(x: int) -> int:
    return int(np.floor(x ** (x / 3) * np.log(scipy.special.factorial(x)) + x))
    # return int(scipy.special.factorial(x))

def flatten_datapoint(matrix, static_labels):
    # Flatten each row in the matrix
    flattened_matrix = matrix.flatten()

    # Convert static labels to a numpy array
    static_labels_array = np.array(static_labels)

    # Append static labels to the flattened matrix
    flattened_datapoint = np.concatenate((flattened_matrix, static_labels_array))

    return flattened_datapoint


def calculate_rank_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure y_true and y_pred are 2-dimensional arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    # Calculate the accuracy for each data point based on how many ranks match
    matches = np.sum(y_true == y_pred, axis=1)
    accuracy = matches / y_true.shape[1]
    return accuracy


def handle_permutations(datapoints, targets, static_labels, k):
    augmented_datapoints, augmented_targets, perm_indices = make_permutations(datapoints, targets, k)
    X = np.array([flatten_datapoint(datapoint, static_labels) for datapoint in augmented_datapoints])
    y = np.array(augmented_targets)
    return X, y, perm_indices





def calculate_average_ranks(y_pred_augmented, perm_indices, num_samples, num_permutations, num_features, order_ranking=True):
    y_pred_augmented = y_pred_augmented.reshape(num_samples, num_permutations, num_features)
    ranks_augmented = np.zeros_like(y_pred_augmented, dtype=int)

    for j in range(y_pred_augmented.shape[0]):
        for p in range(y_pred_augmented.shape[1]):
            if order_ranking:
                ranks_augmented[j, p, :] = np.argsort(np.argsort(y_pred_augmented[j, p, :])) + 1
    if not order_ranking:
        ranks_augmented = y_pred_augmented
    avg_ranks = np.zeros((num_samples, num_features))
    for j in range(num_samples):
        for f in range(num_features):
            rank_sum = 0
            for p in range(num_permutations):
                perm = perm_indices[j * num_permutations + p]
                original_position = perm.index(f)
                rank_sum += ranks_augmented[j, p, original_position]
            if order_ranking:
                avg_ranks[j, f] = int(np.round(rank_sum / num_permutations))
            else:
                avg_ranks[j, f] = rank_sum / num_permutations
    if order_ranking:
        final_ranks = np.zeros((num_samples, num_features), dtype=int)
        # handle ties
        for j in range(num_samples):
            final_ranks[j, :] = rankdata(avg_ranks[j, :], method='ordinal')

        return final_ranks
    return avg_ranks


BEST_DIS = {
    6: {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 70, 'reg_alpha': 0.01, 'reg_lambda': 2, 'subsample': 1.0},
    7: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 50, 'reg_alpha': 1, 'reg_lambda': 0.1, 'subsample': 1.0}
}

def get_training_split(df_of_size, i):
    datapoints, targets, static_labels = vectorize_result_df(df_of_size, i)
    # Split into train+validation and test sets
    datapoints_train_val, datapoints_test, targets_train_val, targets_test = train_test_split(
        datapoints, targets, test_size=0.2, random_state=42069)

    # Further split train+validation into train and validation sets
    datapoints_train, datapoints_val, targets_train, targets_val = train_test_split(
        datapoints_train_val, targets_train_val, test_size=0.25, random_state=42069)  # 0.25 * 0.8 = 0.2
    return datapoints_train, targets_train, datapoints_val, targets_val, datapoints_test, targets_test, static_labels



def run_inference(d, i, X, static_features, rank_order=True):
    k = k_func(i)
    single_datapoint = np.expand_dims(X, axis=0)
    targets_dummy = np.array([np.zeros(i)])  # dummy targets, not required for inference
    X_permutated, _, perm_indices = handle_permutations(single_datapoint, targets_dummy, static_features, k)
    y_pred = d.predict(X_permutated)
    avg_ranks = calculate_average_ranks(y_pred, perm_indices, 1, k, y_pred.shape[1], rank_order)[0]
    return avg_ranks

def get_d_i(i: int):
    df = get_df()
    if i in BEST_DIS:
        df_of_size = partition_by_size(df, i)
        k = k_func(i)

        datapoints, targets, static_labels = vectorize_result_df(df_of_size, i)

        # Split into train and test sets
        datapoints_train, datapoints_test, targets_train, targets_test = train_test_split(datapoints, targets,
                                                                                          test_size=0.3,
                                                                                          random_state=42069)

        # Combine train and validation sets for final training
        X_train, y_train, _ = handle_permutations(datapoints_train, targets_train, static_labels, k)

        # Train the final decision tree model
        xgb = XGBRegressor(**BEST_DIS[i], n_jobs=-1)
        multi_target_xgb = MultiOutputRegressor(xgb)
        multi_target_xgb.fit(X_train, y_train)

        return multi_target_xgb
def get_vectorized_df_of_size(df, size: int):
    df_of_size = partition_by_size(df, size)
    return vectorize_result_df(df_of_size, size)


def get_trained_d_i_on_dataset(i: int, datapoints, targets, static_labels):
    k = k_func(i)
    X_train, y_train, _ = handle_permutations(datapoints, targets, static_labels, k)
    xgb = XGBRegressor(**BEST_DIS[i], n_jobs=-1)
    multi_target_xgb = MultiOutputRegressor(xgb)
    multi_target_xgb.fit(X_train, y_train)
    return multi_target_xgb
def train_d_i(i: int):
    df = get_df()
    df_of_size = partition_by_size(df, i)
    k = k_func(i)
    print(f'Num Permutations {k}', flush=True)

    datapoints, targets, static_labels = vectorize_result_df(df_of_size, i)

    # Split into train and validation sets
    datapoints_train, datapoints_test, targets_train, targets_test = train_test_split(datapoints, targets, test_size=0.3,
                                                                                    random_state=42069)

    kf = KFold(n_splits=5, shuffle=True, random_state=42069)

    param_grid = {
        'n_estimators': [30, 50, 75, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2]
    }

    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for train_index, val_index in kf.split(datapoints_train):
            fold_datapoints_train, fold_datapoints_val = datapoints_train[train_index], datapoints_train[val_index]
            fold_targets_train, fold_targets_val = targets_train[train_index], targets_train[val_index]

            X_train, y_train, _ = handle_permutations(fold_datapoints_train, fold_targets_train, static_labels, k)
            X_val, y_val, perm_indices_val = handle_permutations(fold_datapoints_val, fold_targets_val, static_labels,
                                                                 k)

            xgb = XGBRegressor(**params, n_jobs=-1)
            multi_target_xgb = MultiOutputRegressor(xgb)
            multi_target_xgb.fit(X_train, y_train)

            y_pred_augmented = multi_target_xgb.predict(X_val)
            y_targets = np.array(fold_targets_val)
            avg_ranks = calculate_average_ranks(y_pred_augmented, perm_indices_val, len(fold_datapoints_val), k,
                                                y_pred_augmented.shape[1])
            rank_accuracy = calculate_rank_accuracy(y_targets, avg_ranks)
            fold_scores.append(rank_accuracy)

        mean_fold_score = np.mean(fold_scores)

        if mean_fold_score > best_score:
            # have to take average of folds
            print(f'Best now is: {params} | {mean_fold_score}', flush=True)
            best_score = mean_fold_score
            best_params = params

    print(f'Best Params: {best_params} | Size: {i}', flush=True)
    print(f'Best Validation Rank-Based Accuracy: {best_score} | Size: {i}', flush=True)

    # final_xgb = XGBRegressor(**best_params)
    # final_model = MultiOutputRegressor(final_xgb)
    # final_model.fit(datapoints_train, targets_train)
    #
    # return final_model, best_score


def test_inference(df):
    # test decision tree for size 6
    i = 6
    # get partitioned dataset of size i
    df_of_size = partition_by_size(df, i)
    datapoints, targets, static_labels = vectorize_result_df(df_of_size, i)
    datapoints_train, datapoints_test, targets_train, targets_test = train_test_split(datapoints, targets,
                                                                                          test_size=0.3,
                                                                                          random_state=42069)

    d = get_d_i(i)
    # test first datapoint
    X = datapoints_train[0]
    train_Y = targets_train[0]
    # get flattened test datapoint (for decision tree input)
    print(f' model inference: {run_inference(d, i, X, static_labels)}')
    print(f' actual: {train_Y}')

def report_stats(actual_results, predicted_results):
    N = len(actual_results)

    predict_1st_place = 0
    predict_one_in_top_3 = 0
    predict_entire_top_3 = 0
    predict_all = 0
    predict_at_least_1 = 0
    avg_correct = 0

    for actual_Y, predicted_Y in zip(actual_results, predicted_results):
        # Check if predicted 1st place is correct
        if 1 in predicted_Y and 1 in actual_Y:
            if predicted_Y.index(1) == actual_Y.index(1):
                predict_1st_place += 1

        # Check if any racer in the actual top 3 is correctly predicted in the top 3
        predicted_top_3 = set()
        for pos in range(1, 4):
            if pos in predicted_Y:
                predicted_top_3.add(predicted_Y.index(pos))

        top_3_matches = [actual_Y[index] == predicted_Y[index] for index in predicted_top_3]
        if any(top_3_matches):
            predict_one_in_top_3 += 1

        if all(top_3_matches):
            predict_entire_top_3 += 1

        # Check if the entire ranking is correct
        if predicted_Y == actual_Y:
            predict_all += 1

        # Check if at least one ranking is correct
        if any(predicted_Y[r] == actual_Y[r] for r in range(len(actual_Y))):
            predict_at_least_1 += 1

        avg_correct += sum([predicted_Y[r] == actual_Y[r] for r in range(len(actual_Y))])

    print(f'Predict 1st place %: {predict_1st_place / N * 100:.2f}%')
    print(f'Predict one in top 3 %: {predict_one_in_top_3 / N * 100:.2f}%')
    print(f'Predict entire top 3 %: {predict_entire_top_3 / N * 100:.2f}%')
    print(f'Predict all %: {predict_all / N * 100:.2f}%')
    print(f'Predict at least 1 %: {predict_at_least_1 / N * 100:.2f}%')

def run_training_stats(df, size):
    i = size
    # Get partitioned dataset of size i
    df_of_size = partition_by_size(df, i)
    datapoints, targets, static_labels = vectorize_result_df(df_of_size, i)
    datapoints_train, _, targets_train, _ = train_test_split(datapoints, targets, test_size=0.3, random_state=42069)

    d = get_d_i(i)
    # Record stats for metrics
    N = len(datapoints_train)

    actual_results = []
    predicted_results = []

    for j in range(N):
        X = datapoints_train[j]
        actual_Y = targets_train[j]
        predicted_Y = run_inference(d, i, X, static_labels)

        # Convert to list if not already
        actual_Y = list(actual_Y)
        predicted_Y = list(predicted_Y)

        actual_results.append(actual_Y)
        predicted_results.append(predicted_Y)

    # Call the generalized function for stats reporting
    report_stats(actual_results, predicted_results)

if __name__ == '__main__':
    # RACE_SIZES = [6, 7, 8, 9, 10]
    RACE_SIZES = [6, 7, 8, 9, 10]
    # read in the dataframe
    #
    train_normal_d_i(7)
    # run_training_stats(get_df(), 7)

    # train_d_i(6)
    # test_inference(df)
   #  for size in RACE_SIZES:
   #      train_d_i(df, size)
