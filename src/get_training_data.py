import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoder(df, column, items=None):
    if items is None:
        return single_column_to_one_hot(df, column)
    else:
        categories = {items[i]: i for i in range(len(items))}
        vector_length = len(items)

    def encode_value(value):
        vector = [0] * vector_length
        vector[categories[value]] = 1
        return vector

    df[column] = df[column].apply(encode_value)

    return df


def one_hot_encoder_multi(df, column, items=None):
    categories = {items[i]: i for i in range(len(items))}
    vector_length = len(items)

    def encode_value(value):
        vector = [0] * vector_length
        if value == 'None':
            vector[categories['None']] = 1
            return vector

        for char in value:
            vector[categories[char]] = 1
        return vector

    df[column] = df[column].apply(encode_value)

    return df


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


def vectorize_dfs(df_pp, df_result, race_sizes):
    unique_values = {
        'meds': ['L', 'BL', 'None'],
        'equip': ['None', 'b', 'f', 'h', 'y', 'n', 'r', 'v', 'o', 'g', 'z', 'w'],
        'track_condition': ['HY', 'MY', 'GD', 'FT', 'ST', 'WF', 'SF', 'SL', 'GS', 'YL', 'FM', 'GF', 'SY'],
        'weather': ['E', 'H', 'R', 'F', 'None', 'L', 'S', 'O', 'C']
    }

    def vectorize_df_pp(df_pp):
        # may need to drop 'sex' and 'scratched' column, not present in result
        one_hot_columns = ['sex', 'track_condition', 'weather', 'equip', 'meds', 'scratched']
        date_columns = ['race_date']

        # need to add a fix for equip columns, change multi character
        for column in one_hot_columns:
            if column == 'equip':
                df_pp = one_hot_encoder_multi(df_pp, column, unique_values['equip'])
            elif column in unique_values:
                df_pp = one_hot_encoder(df_pp, column, unique_values[column])
            else:
                df_pp = single_column_to_one_hot(df_pp, column)

        for column in date_columns:
            df_pp = vectorize_date(df_pp, column)

        return df_pp

    def vectorize_df_result(df_result, size):
        # columns in pp not present in result: surface
        one_hot_columns = ['weather', 'surface', 'track_condition']
        date_columns = ['race_date']
        racer_one_hot_columns = ['meds', 'equip']

        for i in range(size):
            for j in racer_one_hot_columns:
                one_hot_columns.append(f'racer_{i}_' + j)

        for column in one_hot_columns:
            trim_col = column.split('_')[-1]
            if trim_col == 'equip':
                df_result = one_hot_encoder_multi(df_result, column, unique_values['equip'])
            elif column in unique_values:
                df_result = one_hot_encoder(df_result, column, unique_values[column])
            elif trim_col in unique_values:
                df_result = one_hot_encoder(df_result, column, unique_values[trim_col])
            else:
                df_result = single_column_to_one_hot(df_result, column)

        for column in date_columns:
            df_result = vectorize_date(df_result, column)

        return df_result

    d = {'df_pp': vectorize_df_pp(df_pp)}
    for i in race_sizes:
        df_result_i = df_result[df_result['num_of_racers'] == i]
        d[f'df_result_{i}'] = vectorize_df_result(df_result_i, i)

    return d


def get_training_data_x(df, columns):
    """
    :param df: ASSUMES df IS ALREADY VECTORIZED
    :param columns: what columns to include in the training data for x
    :return:
    """
    final_matrix = []
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
    return final_matrix


if __name__ == '__main__':
    race_sizes = [6, 7, 8, 9, 10]

    df_pp = pd.read_csv('../data/output_pp.csv', dtype=str)
    df_result = pd.read_csv('../data/output_results_new.csv', dtype=str)
    df_result['num_of_racers'] = df_result['num_of_racers'].astype(int)

    columns_pp = [col for col in df_pp.columns if col != 'race_comments']

    vectorized = vectorize_dfs(df_pp, df_result, race_sizes)
    # X_matrix = get_training_data_x(vectorized['df_pp'], columns_pp)
    # print(X_matrix[0])

    for i in race_sizes:
        df_result_i = vectorized[f'df_result_{i}']
        columns_result = [col for col in df_result.columns[:8] if col != 'number']
        for j in range(i):
            columns_result += [f'racer_{j}_equip', f'racer_{j}_weight', f'racer_{j}_post_position',
                               f'racer_{j}_age', f'racer_{j}_jockey_key', f'racer_{j}_trainer_key',
                               f'racer_{j}_meds', f'racer_{j}_dollar_odds']
        dummy = get_training_data_x(df_result_i, columns_result)
        print(dummy[0])
