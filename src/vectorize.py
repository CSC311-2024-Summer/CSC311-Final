import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
# from dotenv import load_dotenv
# import openai
# from openai.error import RateLimitError, OpenAIError
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
matplotlib.use('Agg')

# load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = ""
# MESSAGE = {
#     {"role": "system", "content": ""},
#     {"role": "user", "content": ""},
# }


def get_gpt_response(messages, max_tries=10, wait_time=10):
    for _ in range(max_tries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            reply = response['choices'][0]['message']['content']
            return reply
        except RateLimitError:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except OpenAIError as e:
            print(f"An error occurred: {e}")
            break
    else:
        raise Exception("Rate limit exceeded. Try again later.")


def parse_comments_in_csv(csv_file, column, messages):
    df = pd.read_csv(csv_file)


def single_column_to_one_hot(df, column):
    encoder = OneHotEncoder(sparse_output=False)
    transformed_data = encoder.fit_transform(df[[column]])
    df[column] = transformed_data.tolist()

    # print(f"One-hot encoded column for '{column}':")
    # print(df[[column]].head())

    return df


def vectorize_date(df, column):
    df[column] = pd.to_datetime(df[column])

    df[column] = df[column].apply(lambda x: [x.year, x.month, x.day])
    return df

def get_pp_Y(df):
    y_vector = df['finish_time'].apply(float).values
    return y_vector

def get_pp_X(df):
    # skip comments column for now
    one_hot_columns = ['sex', 'track_condition', 'weather', 'equipment', 'meds', 'scratched', 'jockey', 'trainer', 'owner']
    date_columns = ['race_date']

    for column in one_hot_columns:
        df = single_column_to_one_hot(df, column)

    for column in date_columns:
        df = vectorize_date(df, column)

    # I still need y vector that retursn all finish_times
    final_matrix = []
    columns = [col for col in df.columns if col not in ['race_comments', 'finish_time', 'reg_num']]
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
                    print(column)
                    print(index)
                    break
        if flag:
            final_matrix.append(row_vector)

    final_matrix = np.array(final_matrix)
    print(final_matrix.shape)
    return final_matrix



def get_training_data_split():
    df = pd.read_csv('../data/output_pp.csv')
    X = get_pp_X(df)
    y = get_pp_Y(df)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42069)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42069)
    return X_train, X_val, y_train, y_val


REGRESSION_PARAM_GRID = {
    'poly__degree': [1, 2, 3, 4, 5],
    'ridge__alpha': [0.2, 0.5, 1, 2, 5]
}


def train_regression_model():
    X_train, X_val, y_train, y_val = get_training_data_split()
    results = []

    for degree in REGRESSION_PARAM_GRID['poly__degree']:
        for alpha in REGRESSION_PARAM_GRID['ridge__alpha']:
            pipeline = Pipeline([
                ('pca', PCA(n_components=0.80)),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('ridge', Ridge(alpha=alpha)),
            ])

            pipeline.fit(X_train, y_train)
            y_val_pred = pipeline.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            results.append((degree, alpha, val_mse, val_r2))
            print(f'Degree: {degree}, Alpha: {alpha}, MSE: {val_mse:.4f}, R_2: {val_r2:.4f}')

    # get the best model based on MSE
    best_degree, best_alpha, best_mse, best_r2 = min(results, key=lambda x: x[2])

    best_model = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
        ('ridge', Ridge(alpha=best_alpha))
    ])

    best_model.fit(X_train, y_train)

    print(f'Best Parameters: Degree: {best_degree}, Alpha: {best_alpha}')
    print(f'Best Validation Mean Squared Error: {best_mse}')
    print(f'Best Validation R_2 Score: {best_r2}')


def visualize_data():
    df = pd.read_csv('../data/output_pp.csv')
    # remove id 'jockey' and 'owner' columns
    df = df.drop(columns=['jockey', 'owner'])
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


# visualize_data()

# Run the function to train the model
train_regression_model()
