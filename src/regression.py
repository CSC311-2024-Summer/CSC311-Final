import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

MODEL_PATH = 'models/regression_model.pkl'
matplotlib.use('Agg')

REGRESSION_PARAM_GRID = {
    'poly__degree': [2],
    'alpha': np.linspace(0.1, 5, 10)  # Alphas from 0 to 5 with 0.25 increment
}

BEST_CONFIG = {'poly__degree': 2, 'alpha': 1.6}

MODELS_FOLDER = 'models'
MODEL_PATH = os.path.join(MODELS_FOLDER, 'regression_model.pkl')
ENCODERS_PATH = os.path.join(MODELS_FOLDER, 'encoders.pkl')


def preprocess_column_for_encoding(df, column):
    # Ensure the column is of string type
    df[column] = df[column].astype(str)

    # Handle missing values if any
    df[column] = df[column].fillna('missing')

    return df


def single_column_to_one_hot(df, column, should_fit=False, encoders_path=ENCODERS_PATH):
    df = preprocess_column_for_encoding(df, column)
    encoders = load_encoders(encoders_path)
    encoder = encoders.get(column, OneHotEncoder(sparse_output=False))
    # ignore OOV
    encoder.handle_unknown = 'ignore'

    if should_fit or column not in encoders:
        # print(f'Refitting Encoder for column: {column}')
        transformed_data = encoder.fit_transform(df[[column]])
    else:
        # print(f"Length of the encoder vocabulary for column '{column}': {len(encoder.categories_[0])}")
        transformed_data = encoder.transform(df[[column]])

    df[column] = transformed_data.tolist()
    encoders[column] = encoder

    if should_fit:
        save_encoders(encoders, encoders_path)

    return df, encoders


def vectorize_date(df, column):
    df[column] = pd.to_datetime(df[column])

    # extract month
    df[f'{column}_month'] = df[column].dt.month

    # drop the original date column
    df = df.drop(columns=[column])

    return df


def encode_month_cyclical(df, column):
    df[column + '_sin'] = np.sin(2 * np.pi * df[column] / 12)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column] / 12)
    return df


def get_pp_Y(df):
    y_vector = df['finish_time'].apply(float).values
    return y_vector


def get_pp_X(df, should_save_encoders=False, encoders_path=ENCODERS_PATH):
    one_hot_columns = ['sex', 'track_condition', 'weather', 'equip', 'meds', 'jockey_key', 'trainer_key',
                       'post_position', 'race_date_month', 'surface']
    date_columns = ['race_date']

    for column in date_columns:
        df = vectorize_date(df, column)
    encoders = None
    for column in one_hot_columns:
        df, encoders = single_column_to_one_hot(df, column, should_save_encoders, encoders_path)

    final_matrix = []
    columns = [col for col in df.columns if
               col not in ['race_comments', 'finish_time', 'reg_num', 'temperature', 'avg_official_finish', 'race_date',
                           'owner_key']]
    numerical_columns = ['dollar_odds', 'weight', 'distance', 'run_up_distance']

    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = np.log1p(df[column])

    for index, row in df.iterrows():
        row_vector = []
        flag = True
        for column in columns:
            if isinstance(row[column], list):
                row_vector.extend(row[column])
            else:
                try:
                    row_vector.append(float(row[column]))
                except ValueError:
                    flag = False
                    break
        if flag:
            final_matrix.append(row_vector)
    final_matrix = np.array(final_matrix)
    if should_save_encoders:
        save_encoders(encoders, encoders_path)
    return final_matrix


def save_encoders(encoders, path=ENCODERS_PATH):
    print('Saved Encoders')
    joblib.dump(encoders, path)


def load_encoders(path=ENCODERS_PATH):
    if os.path.exists(path):
        # print("Loaded Encoders")
        return joblib.load(path)
    return {}


def detect_large_differences():
    df = pd.read_csv('../data/output_pp_new.csv')
    X = get_pp_X(df)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    threshold = np.percentile(stds, 95)
    large_diff_indices = np.where(stds > threshold)[0]

    column_names = [col for col in df.columns if
                    col not in ['race_comments', 'finish_time', 'reg_num', 'temperature', 'avg_official_finish']]
    large_diff_columns = []

    for i in large_diff_indices:
        if i < len(column_names):
            large_diff_columns.append(column_names[i])
        else:
            print(f"Index {i} out of range for column names")

    print("Features with large differences:")
    for col in large_diff_columns:
        print(col)

    return large_diff_columns


def get_training_data_split():
    df = pd.read_csv('../data/output_pp_new.csv')
    X = get_pp_X(df, should_save_encoders=True)
    y = get_pp_Y(df)
    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.2, random_state=42069, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42069,
                                                      shuffle=True)
    return X_train, X_val, y_train, y_val


def get_test_set():
    df = pd.read_csv('../data/output_pp_new.csv')
    X = get_pp_X(df)
    y = get_pp_Y(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42069, shuffle=True)
    return X_test, y_test


def train_and_plot_regression_model():
    X_train, X_val, y_train, y_val = get_training_data_split()
    results = []

    X_train_sparse = csr_matrix(X_train)
    X_val_sparse = csr_matrix(X_val)

    plt.figure(figsize=(14, 7))

    colors = ['b', 'g', 'r', 'c']

    for idx, degree in enumerate(REGRESSION_PARAM_GRID['poly__degree']):
        alphas = REGRESSION_PARAM_GRID['alpha']
        val_mses = []
        val_r2s = []

        for alpha in alphas:
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('ridge', Ridge(alpha=alpha, random_state=42069, solver='sparse_cg'))
            ])

            pipeline.fit(X_train_sparse, y_train)
            y_val_pred = pipeline.predict(X_val_sparse)

            val_mse = mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            val_mses.append(val_mse)
            val_r2s.append(val_r2)
            results.append((degree, alpha, val_mse, val_r2))
            print(f'Degree: {degree}, Alpha: {alpha:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}')

        plt.subplot(1, 2, 1)
        plt.plot(alphas, val_mses, marker='o', color=colors[idx], label=f'Degree {degree}')
        plt.xlabel('Alpha')
        plt.ylabel('Validation MSE')
        plt.title('MSE for Polynomial Degrees')

        plt.subplot(1, 2, 2)
        plt.plot(alphas, val_r2s, marker='o', color=colors[idx], label=f'Degree {degree}')
        plt.xlabel('Alpha')
        plt.ylabel('Validation R²')
        plt.title('R² for Different Polynomial Degrees')

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_polynomial_degrees.png')
    plt.close()

    best_degree, best_alpha, best_mse, best_r2 = min(results, key=lambda x: x[2])

    best_model = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
        ('sgd', Ridge(alpha=best_alpha))
    ])

    best_model.fit(X_train_sparse, y_train)

    print(f'Best Parameters: Degree: {best_degree}, Alpha: {best_alpha}')
    print(f'Best Validation Mean Squared Error: {best_mse:.4f}')
    print(f'Best Validation R² Score: {best_r2:.4f}')


def regression_best_param_test():
    best_param = {'degree': 2, 'alpha': 1.73}
    X_train, _, y_train, _ = get_training_data_split()
    X_test, y_test = get_test_set()
    X_test_sparse = csr_matrix(X_test)
    X_train_sparse = csr_matrix(X_train)
    best_model = Pipeline([
        ('poly', PolynomialFeatures(degree=best_param['degree'], include_bias=False)),
        ('sgd', Ridge(alpha=best_param['alpha'], solver='sparse_cg'))
    ])

    best_model.fit(X_train_sparse, y_train)
    y_pred = best_model.predict(X_test_sparse)
    val_mse = mean_squared_error(y_pred, y_test)
    print(f'Test Mean Squared Error: {val_mse:.4f}')


def visualize_data():
    df = pd.read_csv('../data/output_pp_new.csv')
    # remove id 'jockey' and 'owner' columns
    df = df.drop(columns=['jockey_key'])
    # calculate some important stats
    summary_stats = df.describe().T
    summary_stats['range'] = summary_stats['max'] - summary_stats['min']
    summary_stats['std_dev'] = summary_stats['std']
    summary_stats = summary_stats[['mean', 'std_dev', 'min', '25%', '50%', '75%', 'max', 'range']]

    # summary statistics
    print("Summary Statistics:")
    print(summary_stats)

    # plot histograms and save them
    df.hist(bins=50, figsize=(20, 15))
    plt.suptitle('Feature Distributions', fontsize=20)
    plt.savefig('feature_distributions.png')
    plt.close()

    # label axis
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # plot boxplots to identify outliers visually and save them
    df.plot(kind='box', subplots=True, layout=(len(df.columns) // 4 + 1, 4), figsize=(20, 15), sharex=False,
            sharey=False)
    plt.suptitle('Boxplots for Feature Distributions', fontsize=20)
    plt.savefig('boxplots.png')
    plt.close()


def visulization_correlation():
    df = pd.read_csv('../data/output_pp_new.csv')
    get_pp_X(df)
    non_cont_columns = ['sex', 'track_condition', 'weather', 'equip', 'meds', 'jockey_key', 'trainer_key', 'reg_num',
                        'race_date', 'race_comments']
    df = df.drop(columns=non_cont_columns)

    # correlation analysis
    corr_matrix = df.corr()

    finish_time_corr = corr_matrix[['finish_time']].sort_values(by='finish_time', ascending=False)

    plt.figure(figsize=(8, 10))
    heatmap = sns.heatmap(finish_time_corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    plt.title('Correlation with Finish Time', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_with_finish_time.png')
    plt.close()


def run_inference(model, x):
    x_sparse = csr_matrix(x)
    return model.predict(x_sparse)


def run_inference_on_race(model, x, r):
    pass


def get_regression_model():
    if os.path.exists(MODEL_PATH):
        # load the model if it exists
        print('Loaded Regression Model From Cache')
        pipeline = joblib.load(MODEL_PATH)
    else:
        print('Creating Regression Model')
        # train and save
        X_train, _, y_train, _ = get_training_data_split()
        X_train_sparse = csr_matrix(X_train)
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=BEST_CONFIG['poly__degree'], include_bias=False)),
            ('ridge', Ridge(alpha=BEST_CONFIG['alpha'], random_state=42069, solver='sparse_cg'))
        ])
        pipeline.fit(X_train_sparse, y_train)
        # save the model
        joblib.dump(pipeline, MODEL_PATH)

    return pipeline
