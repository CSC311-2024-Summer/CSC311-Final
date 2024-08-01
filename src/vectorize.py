import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import openai
from openai.error import RateLimitError, OpenAIError
from sklearn.preprocessing import OneHotEncoder

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

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


def vectorize_pp_df(df):
    # skip comments column for now
    one_hot_columns = ['sex', 'track_condition', 'weather', 'equipment', 'meds', 'scratched']
    date_columns = ['race_date']

    for column in one_hot_columns:
        df = single_column_to_one_hot(df, column)

    for column in date_columns:
        df = vectorize_date(df, column)

    final_matrix = []
    columns = [col for col in df.columns if col != 'race_comments']
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
    df = pd.read_csv('../data/output_pp.csv')
    matrix = vectorize_pp_df(df)
    print(matrix)
