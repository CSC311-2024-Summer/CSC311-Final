import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def partition_by_size(df, size):
    """
    :param df: master dataset
    :param size:
    :return: new df with only race size = size
    """
    return df[df['num_of_racers'] == size]


def single_column_to_one_hot(df, column):
    try:
        encoder = OneHotEncoder(sparse_output=False)
        transformed_data = encoder.fit_transform(df[[column]])
        df[column] = transformed_data.tolist()
        # df[column] = [list(row) for row in transformed_data]
    except TypeError as e:
        # print(e)
        # print(column)
        # Debugging: Identify non-string entries
        non_string_entries = df[df[column].apply(lambda x: not isinstance(x, str))][column]
        if not non_string_entries.empty:
            print(f"Non-string entries found in '{column}' column:")
            print(non_string_entries)

    return df


def vectorize_date(df, column):
    df[column] = pd.to_datetime(df[column]).dt.date
    df[column] = df[column].apply(lambda x: [x.year, x.month, x.day])

    print(f"Vectorized '{column}':")
    print(df[[column]].head())

    return df


def vectorize_last_pp(df, column):
    ...


def vectorize_result_df(df, size):
    one_hot_columns = ['weather', 'surface', 'track_condition']
    racer_one_hot_columns = ['meds', 'equip']
    date_columns = ['date']

    for i in range(size):
        for j in racer_one_hot_columns:
            one_hot_columns.append(f'racer_{i}_' + j)

    for column in one_hot_columns:
        df = single_column_to_one_hot(df, column)

    for column in date_columns:
        df = vectorize_date(df, column)

    final_matrix = []
    columns = list(df.columns)[:8] + one_hot_columns[3:]
    print('inside vectorize function:', df.shape)
    for index, row in df.iterrows():
        row_vector = []
        flag = True
        for column in columns:
            if isinstance(row[column], list):
                row_vector.extend(row[cox`lumn])
            else:
                try:
                    row_vector.append(float(row[column]))
                except ValueError:
                    flag = False
                    break
        if flag:
            final_matrix.append(row_vector)

    final_matrix = np.array(final_matrix)
    return final_matrix


if __name__ == '__main__':
    RACE_SIZES = [6, 7, 8, 9, 10]
    df = pd.read_csv('../data/output_results.csv', dtype=str)
    df['num_of_racers'] = df['num_of_racers'].astype(int)

    for size in RACE_SIZES:
        df_of_size = partition_by_size(df, size)
        print(df_of_size.shape)
        matrix = vectorize_result_df(df_of_size, size)
        print(matrix)
        print(matrix[0])
        print('=================')
