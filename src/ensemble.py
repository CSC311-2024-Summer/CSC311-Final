import random

import numpy as np
import pandas as pd
import scipy
import scipy.special
import seaborn as sns
from scipy.stats import rankdata
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
import os
import pickle
import regression
import decision_tree
import gradient_descent as gd
import itertools
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

REGRESSION_COLUMN_NAMES = ['sex', 'age', 'num_of_racers', 'race_date', 'run_up_distance', 'track_condition', 'distance', 'weather', 'post_position', 'equip', 'jockey_key', 'weight', 'meds', 'dollar_odds', 'trainer_key', 'surface']


random.seed(42069)

def build_racer_columns(size):
    # values we want to extract
    columns_to_build = [('sex', True), ('age', True), size, ('race_date', False), ('run_up_distance', False),
                        ('track_condition', False), ('distance', False), ('weather', False), ('post_position', True),
                        ('equip', True),
                        ('jockey_key', True), ('weight', True), ('meds', True), ('dollar_odds', True),
                        ('trainer_key', True), ('surface', False)]

    grouped_columns = []
    for i in range(size):
        racer_columns = []
        for col in columns_to_build:
            if isinstance(col, tuple):
                if col[1]:
                    racer_columns.append(f'racer_{i}_{col[0]}')
                else:
                    racer_columns.append(col[0])
            else:
                racer_columns.append(col)
        grouped_columns.append(racer_columns)
    return grouped_columns


def extract_racer_data(df, grouped_columns):
    extracted_data = []

    for index, row in df.iterrows():
        row_data = []
        for columns in grouped_columns:
            row_values = []
            for col in columns:
                if col in df.columns:
                    row_values.append(row[col])
                else:
                    row_values.append(col)
            row_data.append(row_values)
        extracted_data.append(row_data)

    return extracted_data



def get_datasets_from_df(df, size):
    """
    parses results dataset for both regression model and decision tree
    :param df_of_size:
    :param size:
    :return:
    """
    # parse for regression model
    regression_columns = build_racer_columns(size)
    regression_data = extract_racer_data(df, regression_columns)
    # convert to df for regression model
    regression_dfs = [pd.DataFrame(regression_data[i], columns=REGRESSION_COLUMN_NAMES) for i in range(len(regression_data))]
    # build dataset
    regression_data = [regression.get_pp_X(df, should_save_encoders=False) for df in regression_dfs]
    decision_tree_data, target_labels, static_labels = decision_tree.get_vectorized_df_of_size(df, size)
    # # Process racer-specific columns
    # for group in grouped_racer_columns:
    #     df = process_label_columns(df, [col for col in group if any(label in col for label in racer_label_columns)])
    #     df = process_numeric_columns(df, [col for col in group if
    #                                       any(numeric in col for numeric in racer_numerical_columns)])
    # df = process_numeric_columns(df, [f"racer_{i}_{target_column}" for i in range(size)])
    return regression_data, decision_tree_data, target_labels, static_labels


def run_inference_both_models(df_datapoint, size, regression_model, dt, should_order_rank=False):
    regression_data, decision_tree_data, target_labels, static_labels = get_datasets_from_df(df_datapoint, size)
    # run inference on regression
    regression_result = []
    # run regression inference
    for x in regression_data[0]:
        regression_result.append(regression.run_inference(regression_model, x.reshape(1, -1))[0])
    if should_order_rank:
        # get ranking if that's what we're looking for
        regression_ranking = (np.argsort(regression_result) + 1).tolist()
    else:
        regression_ranking = regression_result
    # run decision tree inference
    decision_tree_ranking = decision_tree.run_inference(dt, size, decision_tree_data[0], static_labels, should_order_rank)
    target_labels = target_labels[0]
    return regression_ranking, decision_tree_ranking.tolist(), target_labels


def get_cache_path(size):
    return f"oof_predictions_cache_size_{size}.pkl"

def save_to_cache(size, oof_predictions):
    cache_path = get_cache_path(size)
    with open(cache_path, 'wb') as f:
        pickle.dump(oof_predictions, f)

def load_from_cache(size):
    cache_path = get_cache_path(size)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def get_test_split(size):
    df = decision_tree.get_df()
    df_of_size = decision_tree.partition_by_size(df, size)
    _, test_datapoints, _, test_targets = train_test_split(df_of_size, np.ones([len(df_of_size)]), test_size=0.3, random_state=42069)
    return test_datapoints, test_targets

def get_dataset_with_oof_predictions(size, ranking=True):
    # Check if OOF predictions are already cached
    oof_predictions = load_from_cache(size)
    if oof_predictions is not None:
        print("Loaded OOF predictions from cache.")
        df = decision_tree.get_df()
        df_of_size = decision_tree.partition_by_size(df, size)
        return df_of_size, oof_predictions

    # If not cached, calculate OOF predictions
    df = decision_tree.get_df()
    df_of_size = decision_tree.partition_by_size(df, size)
    training_datapoints, _, _, _ = train_test_split(df_of_size, np.ones([len(df_of_size)]), test_size=0.3, random_state=42069)
    kf = KFold(n_splits=5, shuffle=True, random_state=42069)

    oof_predictions = [None] * len(training_datapoints)

    for train_index, val_index in kf.split(training_datapoints):
        datapoints_train, datapoints_val = training_datapoints.iloc[train_index], training_datapoints.iloc[val_index]
        # vectorize training data
        X_train, train_targets, static_labels_train = decision_tree.vectorize_result_df(datapoints_train, size)
        dt_fold_model = decision_tree.get_trained_d_i_on_dataset(size, X_train, train_targets, static_labels_train)\
        # process one at a time since inference can only accept one
        for idx in val_index:
            X_val, _, static_labels_val = decision_tree.vectorize_result_df(training_datapoints.iloc[[idx]], size)
            val_prediction = decision_tree.run_inference(dt_fold_model, size, X_val[0], static_labels_val, ranking)
            oof_predictions[idx] = val_prediction.tolist()

    # Save OOF predictions to cache
    save_to_cache(size, oof_predictions)
    print("OOF predictions saved to cache.")
    print(oof_predictions)
    return df_of_size, oof_predictions



def generate_disagrement_graph_stats(dt_predictions, regression_predictions, nn_predictions, ensemble_predictions):
    predictions_list = [np.array(dt_predictions), np.array(regression_predictions), np.array(nn_predictions),
                        np.array(ensemble_predictions)]

    num_models = len(predictions_list)
    num_data_points = predictions_list[0].shape[0]

    disagreement_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(num_models):
            # Compare all rankings for each pair of models across all data points
            disagreements = [
                np.mean(predictions_list[i][k] != predictions_list[j][k])
                for k in range(num_data_points)
            ]
            disagreement_matrix[i, j] = np.mean(disagreements)

    # Plot the disagreement matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(disagreement_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                xticklabels=["DT", "Regression", "NN", "Ensemble"],
                yticklabels=["DT", "Regression", "NN", "Ensemble"])
    plt.title("Disagreement Matrix of Model Predictions")
    plt.tight_layout()
    plt.savefig('disagreement_matrix_fixed.png')
    plt.show()

def get_nn_inference_stats(targets, dt_predictions, regression_predictions, nn_predictions, display_full_stats=True):
    # convert both to numpy array
    targets = np.array(targets)
    nn_predictions = np.array(nn_predictions)
    N = len(targets)
    dt_ranks = []
    reg_ranks = []
    fractions_correct = np.zeros(N)
    for i in range(N):
        dt_rank = (np.argsort(np.argsort(dt_predictions[i, :])) + 1).tolist()

        # Rank regression predictions
        reg_rank = (np.argsort(np.argsort(regression_predictions[i, :])) + 1).tolist()
        # print(f'ensemble: {ensemble_ranks[i, :]}')
        # print(f'targets: {targets[i]}')
        # print(f'dt: {dt_rank}')
        # print(f'reg: {reg_rank}')
        dt_ranks.append(dt_rank)
        reg_ranks.append(reg_rank)
        # rank
        num_correct = np.sum(targets[i, :] == nn_predictions[i, :])
        fractions_correct[i] = num_correct / nn_predictions.shape[1]
    # back to list
    mean_fraction_correct = np.mean(fractions_correct)
    if display_full_stats:
        targets = targets.tolist()
        nn_predictions = nn_predictions.tolist()
        print(f'Validation Score: {mean_fraction_correct}')
        print('Stats For Regression')
        decision_tree.report_stats(reg_ranks, targets)
        print('Stats For GB forest')
        decision_tree.report_stats(dt_ranks, targets)
        print('Stats For NN')
        decision_tree.report_stats(nn_predictions, targets)
    return mean_fraction_correct

def get_inference_stats(targets, dt_predictions, regression_predictions, alpha, beta):
    ensemble_predictions = alpha * dt_predictions + beta * regression_predictions
    ensemble_ranks = np.zeros_like(ensemble_predictions, dtype=int)
    N = len(targets)
    fractions_correct = np.zeros(N)
    dt_ranks = []
    reg_ranks = []
    for i in range(N):
        ensemble_ranks[i, :] = rankdata(ensemble_predictions[i, :], method='ordinal')
        dt_rank = (np.argsort(np.argsort(dt_predictions[i, :])) + 1).tolist()

        # Rank regression predictions
        reg_rank = (np.argsort(np.argsort(regression_predictions[i, :])) + 1).tolist()
        # print(f'ensemble: {ensemble_ranks[i, :]}')
        # print(f'targets: {targets[i]}')
        # print(f'dt: {dt_rank}')
        # print(f'reg: {reg_rank}')
        dt_ranks.append(dt_rank)
        reg_ranks.append(reg_rank)
        # rank
        num_correct = np.sum(targets[i, :] == ensemble_ranks[i, :])
        fractions_correct[i] = num_correct / ensemble_predictions.shape[1]
    # calculate exact matches
    targets = targets.tolist()
    mean_fraction_correct = np.mean(fractions_correct)
    print(f'Validation Score: {mean_fraction_correct}')
    print('Stats For Regression')
    decision_tree.report_stats(reg_ranks, targets)
    print('Stats For GB forest')
    decision_tree.report_stats(dt_ranks, targets)
    print('Stats For Ensemble')
    decision_tree.report_stats(ensemble_ranks.tolist(), targets)

    # Compute the average exact match rate
    return mean_fraction_correct, ensemble_ranks


def get_weights(race_size, train_oof_preds, train_reg_preds, train_targets, lr, l1, l2, num_iterations=500, grad_clip_threshold=1):
    alpha = np.random.rand(race_size)
    beta = np.random.rand(race_size)
    alpha, beta = gd.gradient_descent(alpha, beta, train_oof_preds, train_reg_preds, train_targets,
                                      lr, l1, l2, grad_clip_threshold, num_iterations)

    return alpha, beta


def get_minmax_normalized_data(a, b):
    min_val = min(a.min(), b.min())
    max_val = max(a.max(), b.max())

    a = (a - min_val) / (max_val - min_val)
    b = (b - min_val) / (max_val - min_val)

    return a, b

def get_minmax_normalized_data_singular(a):
    min_val = a.min()
    max_val = a.max()

    a = (a - min_val) / (max_val - min_val)

    return a


def get_predictions(datapoints, size, regression_model, dt_model, ranking=False):
    # Generate non-OOF predictions and collect targets
    regression_predictions = []
    targets = []
    dt_predictions = []
    for i in range(len(datapoints)):
        datapoint = datapoints.iloc[[i]]
        reg, tree, target = run_inference_both_models(datapoint, size, regression_model, dt_model, ranking)
        targets.append(target)
        regression_predictions.append(reg)
        dt_predictions.append(tree)
    return targets, regression_predictions, dt_predictions

def hinge_rank_loss(predictions, targets, margin=0.1):
    # Calculate differences in predictions
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(2)  # [batch_size, n, n]
    # Calculate differences between targets
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(2)  # [batch_size, n, n]

    # Apply hinge loss: max(0, margin - pred_diff * target_diff)
    loss = torch.clamp(margin - pred_diff * target_diff, min=0)

    # Return mean loss
    return loss.mean()

class MetaLearnerNN(nn.Module):
    def __init__(self, input_size, num_neurons, dropout_rate):
        super(MetaLearnerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, num_neurons)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.output = nn.Linear(num_neurons, 7)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

    def predict_ranks(self, x):
        with torch.no_grad():
            scores = self.forward(x)
            ranks = torch.argsort(scores, dim=1, descending=True) + 1
        return ranks.cpu().numpy().tolist()


def run_inference_nn(size, inputs, learning_rate, num_neurons, dropout_rate, regression_model, decision_tree_model, num_epochs=100):
    # Prepare data
    datapoints, oof_predictions = get_dataset_with_oof_predictions(size, False)
    training_targets, regression_predictions, dt_predictions = get_predictions(
        datapoints, size, regression_model, decision_tree_model, True
    )

    regression_predictions = get_minmax_normalized_data_singular(np.array(regression_predictions))
    oof_predictions, dt_predictions = get_minmax_normalized_data(np.array(oof_predictions), np.array(dt_predictions))

    # Convert lists to numpy arrays and PyTorch tensors
    regression_predictions = torch.tensor(np.array(regression_predictions), dtype=torch.float32)
    oof_predictions = torch.tensor(np.array(oof_predictions), dtype=torch.float32)
    dt_predictions = torch.tensor(np.array(dt_predictions), dtype=torch.float32)
    training_targets = torch.tensor(np.array(training_targets), dtype=torch.float32)

    # First split: 70% training, 30% test
    train_reg_preds, test_reg_preds, train_dt_preds, test_dt_preds, train_targets, test_targets = train_test_split(
        regression_predictions, dt_predictions, training_targets, test_size=0.3, random_state=42069
    )

    # Second split: 80% training, 20% validation
    train_reg_preds, val_reg_preds, train_dt_preds, val_dt_preds, train_targets, val_targets, train_oof_preds, val_oof_preds = train_test_split(
        train_reg_preds, train_dt_preds, train_targets, oof_predictions, test_size=0.2, random_state=42069
    )

    # Combine inputs
    train_inputs = torch.cat((train_oof_preds, train_reg_preds), dim=1)
    val_inputs = torch.cat((val_oof_preds, val_reg_preds), dim=1)

    # Initialize model
    model = MetaLearnerNN(input_size=train_inputs.shape[1], num_neurons=num_neurons, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        train_predictions = model(train_inputs)
        loss = hinge_rank_loss(train_predictions, train_targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_inputs)
        val_loss = hinge_rank_loss(val_predictions, val_targets)

    # run inputs
    nn_ranks = model.predict_ranks(inputs)

    return nn_ranks


def train_ensemble(size, learning_rates, l1s, l2s, plot):
    # best weights
    # for testing purposes to verify full dt_model produces different outputs than oof
    dt_model = decision_tree.get_d_i(size)
    regression_model = regression.get_regression_model()

    # Get data points and OOF predictions
    datapoints, oof_predictions = get_dataset_with_oof_predictions(size, False)

    # Get all predictions
    training_targets, regression_predictions, dt_predictions = get_predictions(datapoints, size, regression_model, dt_model)

    # Convert lists to numpy arrays for easier manipulation
    regression_predictions = np.array(regression_predictions)
    oof_predictions = np.array(oof_predictions)
    dt_predictions = np.array(dt_predictions)
    training_targets = np.array(training_targets)

    # Normalize predictions on the same scale
    regression_predictions = softmax(regression_predictions, axis=1)
    oof_predictions = softmax(oof_predictions, axis=1)
    dt_predictions = softmax(dt_predictions, axis=1)


    # First split: 70% training, 30% test (only on regression predictions, dt predictions, and targets)
    (train_reg_preds, test_reg_preds,
     train_dt_preds, test_dt_preds,
     train_targets, test_targets) = train_test_split(regression_predictions, dt_predictions, training_targets,
                                                     test_size=0.3, random_state=42069)

    # At this point, train_oof_preds is associated with the training set
    train_oof_preds = oof_predictions

    # Second split: 80% training, 20% validation (on the training data from the first split)
    (train_reg_preds, val_reg_preds,
     train_dt_preds, val_dt_preds,
     train_targets, val_targets,
     train_oof_preds, val_oof_preds) = train_test_split(train_reg_preds, train_dt_preds, train_targets,
                                                        train_oof_preds, test_size=0.2, random_state=42069)

    # At thi
    # Initialize weights and hyperparameters

    # Store the hyperparameters and their corresponding costs
    hyperparameter_combinations = []
    costs = []


    alpha = np.random.rand(size) * 0.01
    beta = np.random.rand(size) * 0.01
    grad_clip_threshold = 1.0
    num_iterations = 1000

    best_cost = float('inf')
    cost_history = []
    for lr in learning_rates:
        for l1 in l1s:
            for l2 in l2s:
                alpha, beta = gd.gradient_descent(alpha, beta, train_oof_preds, train_reg_preds, train_targets,
                                                  lr, l1, l2, grad_clip_threshold, num_iterations)
                # check on validation using full dt_predictions
                val_cost = gd.compute_cost(alpha, beta, val_dt_preds, val_reg_preds, val_targets, l1, l2)
                cost_history.append(val_cost)
                print(val_cost)

                # Store the hyperparameters and their cost
                hyperparameter_combinations.append((lr, l1, l2))
                costs.append(val_cost)

                # Check if this is the best model
                if val_cost < best_cost:
                    best_cost = val_cost
                    best_alpha = np.copy(alpha)
                    best_beta = np.copy(beta)
                    best_settings = {'learning_rate': lr, 'lambda_1': l1, 'lambda_2': l2}

    if plot:
        print('Best Hyperparameters:', best_settings)
        print('Best Validation Cost:', best_cost)
        print(f"Weights: {alpha} {beta}")

        # Convert the list of hyperparameter combinations and costs to a numpy array
        hyperparameters = np.array(hyperparameter_combinations)
        costs = np.array(costs)

        # Reshape the costs array to match the grid of hyperparameters
        cost_matrix = costs.reshape((len(learning_rates), len(l1s), len(l2s)))

        # Create heatmaps for each value of l2
        for i, l2_val in enumerate(l2s):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cost_matrix[:, :, i], annot=True, fmt=".4f", cmap='viridis',
                        xticklabels=learning_rates, yticklabels=l1s)
            plt.title(f'Cost for L2={l2_val}')
            plt.xlabel('Learning Rate')
            plt.ylabel('L1')
            plt.savefig(f'cost_heatmap_l2_{l2_val}.png')

    # check validatio
    # for validation set.03638975, -0.10930881, -0.03475271, -0.01657101, -0.08070723, -0.03910439,
    print('VALIDATION')
    get_inference_stats(val_targets, val_oof_preds, val_reg_preds, alpha, beta)
    # test set
    test_datapoints, _ = get_test_split(size)
    test_targets, test_reg_pred, test_dt_pred = get_predictions(test_datapoints, size, regression_model, dt_model)
    # get inference stats for test set
    print('TEST SET')
    _, ensemble_ranks = get_inference_stats(np.array(test_targets), np.array(test_dt_pred), np.array(test_reg_pred), alpha, beta)


    # generate comparison matrix
    # get NN output
    nn_preds = run_inference_nn(7, torch.cat([torch.tensor(np.array(test_dt_pred), dtype=torch.float32), torch.tensor(np.array(test_reg_pred), dtype=torch.float32)], dim=1), 0.0008, 6, 0.05, regression_model, dt_model, 100)
    generate_disagrement_graph_stats(test_dt_pred, test_reg_preds, nn_preds, ensemble_ranks)
    # print(f'Ensemble Validation Score: {validation_score}')
    # print(f'Ensemble Weights: {alpha} {beta}')

def run_validation_test(size):
    df = decision_tree.get_df()
    df_of_size = decision_tree.partition_by_size(df, size)

    # Split into train and test sets
    _, test_set, _, _ = decision_tree.train_test_split(df_of_size, np.ones(len(df_of_size)), test_size=0.2,
                                                                                    random_state=42069)
    N = 0
    tree_tot = 0
    reg_tot = 0
    print(len(test_set))
    for i in range(0, len(test_set)):
        N += 1
        datapoint = test_set.iloc[[i]]
        reg, tree, targets = run_inference_both_models(datapoint, size)
        reg_result = decision_tree.calculate_rank_accuracy(targets, reg)
        tree_result = decision_tree.calculate_rank_accuracy(targets, tree)
        tree_tot += tree_result
        reg_tot += reg_result
        print(f'tree: {tree_result}')
        print(f'reg: {reg_result}')
    print(f'avg tree {tree_tot / N}')
    print(f'avg reg {reg_tot / N}')






# def train_meta_learner(size, learning_rate=0.001, num_epochs=100):
#     dt_model = decision_tree.get_d_i(size)
#     regression_model = regression.get_regression_model()
#
#     # Get data points and OOF predictions
#     datapoints, oof_predictions = get_dataset_with_oof_predictions(size, True)
#
#     # Get all predictions
#     training_targets, regression_predictions, dt_predictions = get_predictions(datapoints, size, regression_model,
#                                                                                dt_model, True)
#    #  training_targets = [arr * 10 for arr in training_targets] # scale to stop small difference abuse
#     # scale using minmax
#     regression_predictions = np.array(regression_predictions)
#     oof_predictions = np.array(oof_predictions)
#     dt_predictions = np.array(dt_predictions)
#     training_targets = np.array(training_targets)
#     # regression_predictions = get_minmax_normalized_data_singular(regression_predictions)
#     # oof_predictions, dt_predictions = get_minmax_normalized_data(oof_predictions, dt_predictions)
#     print(regression_predictions)
#     print(oof_predictions)
#     # Convert lists to numpy arrays and then to PyTorch tensors
#     regression_predictions = torch.tensor(regression_predictions, dtype=torch.float32)
#     oof_predictions = torch.tensor(oof_predictions, dtype=torch.float32)
#     dt_predictions = torch.tensor(dt_predictions, dtype=torch.float32)
#     training_targets = torch.tensor(training_targets, dtype=torch.float32)
#
#     # First split: 70% training, 30% test (only on regression predictions, dt predictions, and targets)
#     (train_reg_preds, test_reg_preds,
#      train_dt_preds, test_dt_preds,
#      train_targets, test_targets) = train_test_split(regression_predictions, dt_predictions, training_targets,
#                                                      test_size=0.3, random_state=42069)
#
#     # train is oof_predictions
#     train_oof_preds = oof_predictions
#
#     # Second split: 80% training, 20% validation (on the training data from the first split)
#     (train_reg_preds, val_reg_preds,
#      train_dt_preds, val_dt_preds,
#      train_targets, val_targets,
#      train_oof_preds, val_oof_preds) = train_test_split(train_reg_preds, train_dt_preds, train_targets,
#                                                         train_oof_preds, test_size=0.2, random_state=42069)
#
#     # Min-max scaling for inputs (oof and regression outputs)
#
#     # Combine the relevant predictions into a single input tensor
#     train_inputs = torch.cat((train_oof_preds, train_reg_preds), dim=1)
#     val_inputs = torch.cat((val_oof_preds, val_reg_preds), dim=1)
#     test_inputs = torch.cat((test_dt_preds, test_reg_preds), dim=1)
#
#     # Initialize the model, loss function, and optimizer
#     model = MetaLearnerNN(input_size=train_inputs.shape[1])
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#
#         train_predictions = model(train_inputs)
#         loss = ranknet_loss(train_predictions, train_targets)
#
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 20 == 0:
#             model.eval()
#             with torch.no_grad():
#                 val_predictions = model(val_inputs)
#                 val_loss = ranknet_loss(val_predictions, val_targets)
#             print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
#
#
#     # test on validation
#     nn_ranks = model.predict_ranks(val_inputs)
#     print('VALIDATION')
#     # convert into list array from tensor
#     val_targets = val_targets.cpu().numpy().tolist()
#     validation_score = get_nn_inference_stats(val_targets, val_oof_preds, val_reg_preds, nn_ranks)
#     # test set
#     # print('TEST')
#     # nn_ranks = model.predict_ranks(test_inputs)
#     # get_nn_inference_stats(np.array(test_targets), np.array(test_dt_preds), np.array(test_reg_preds), nn_ranks)
#
#     # Evaluate on the test set
#     # model.eval()
#     # with torch.no_grad():
#     #     test_predictions = model(test_inputs)
#     #     test_loss = pairwise_ranking_loss(test_predictions, test_targets)
#     # print(f'Test Loss: {test_loss.item()}')
#
#     return model
#
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def train_and_evaluate(size, learning_rate, num_neurons, dropout_rate, regression_model, decision_tree_model, num_epochs=100):
    # Prepare data
    datapoints, oof_predictions = get_dataset_with_oof_predictions(size, False)
    training_targets, regression_predictions, dt_predictions = get_predictions(
        datapoints, size, regression_model, decision_tree_model, True
    )

    regression_predictions = get_minmax_normalized_data_singular(np.array(regression_predictions))
    oof_predictions, dt_predictions = get_minmax_normalized_data(np.array(oof_predictions), np.array(dt_predictions))

    # Convert lists to numpy arrays and PyTorch tensors
    regression_predictions = torch.tensor(np.array(regression_predictions), dtype=torch.float32)
    oof_predictions = torch.tensor(np.array(oof_predictions), dtype=torch.float32)
    dt_predictions = torch.tensor(np.array(dt_predictions), dtype=torch.float32)
    training_targets = torch.tensor(np.array(training_targets), dtype=torch.float32)

    # First split: 70% training, 30% test
    train_reg_preds, test_reg_preds, train_dt_preds, test_dt_preds, train_targets, test_targets = train_test_split(
        regression_predictions, dt_predictions, training_targets, test_size=0.3, random_state=42069
    )

    # Second split: 80% training, 20% validation
    train_reg_preds, val_reg_preds, train_dt_preds, val_dt_preds, train_targets, val_targets, train_oof_preds, val_oof_preds = train_test_split(
        train_reg_preds, train_dt_preds, train_targets, oof_predictions, test_size=0.2, random_state=42069
    )

    # Combine inputs
    train_inputs = torch.cat((train_oof_preds, train_reg_preds), dim=1)
    val_inputs = torch.cat((val_oof_preds, val_reg_preds), dim=1)
    test_inputs = torch.cat((test_dt_preds, test_reg_preds), dim=1)

    # Initialize model
    model = MetaLearnerNN(input_size=train_inputs.shape[1], num_neurons=num_neurons, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        train_predictions = model(train_inputs)
        loss = hinge_rank_loss(train_predictions, train_targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_inputs)
        val_loss = hinge_rank_loss(val_predictions, val_targets)

    # Calculate validation score
    nn_ranks = model.predict_ranks(val_inputs)
    validation_score = get_nn_inference_stats(val_targets.cpu().numpy().tolist(), val_oof_preds, val_reg_preds,
                                              nn_ranks, True)
    # report test
    nn_ranks_test = model.predict_ranks(test_inputs)
    print('TEST RESULTS')
    test_score = get_nn_inference_stats(test_targets.cpu().numpy().tolist(), test_dt_preds, test_reg_preds, nn_ranks_test, True)
    return validation_score

def nn__param_search():
    learning_rates = [0.0008]
    num_neurons_list = [6]
    dropout_rates = [0.05]

    best_score = float('-inf')
    best_config = None
    r = regression.get_regression_model()
    dt = decision_tree.get_d_i(7)

    # Grid search with refined configurations
    for lr, neurons, dropout in itertools.product(learning_rates, num_neurons_list, dropout_rates):
        print(f"Testing configuration: learning_rate={lr}, num_neurons={neurons}, dropout_rate={dropout}")
        score = train_and_evaluate(size=7, learning_rate=lr, num_neurons=neurons, dropout_rate=dropout,
                                   num_epochs=100, regression_model=r, decision_tree_model=dt)
        print(f"Validation Score: {score}")
        if score > best_score:
            best_score = score
            best_config = (lr, neurons, dropout)

    print(
        f"Best configuration: learning_rate={best_config[0]}, num_neurons={best_config[1]}, dropout_rate={best_config[2]}")
    print(f"Best validation score: {best_score}")


if __name__ == '__main__':
    set_seed(42069) # for reproduction purposes
    print(train_ensemble(7, [0.01], [0.01], [0.001], True))
