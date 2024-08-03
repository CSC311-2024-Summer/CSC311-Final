from datetime import datetime

import pandas as pd
import os
import csv

from data import parse_xml_pp


def get_regNum_name_age_pp(directory, output_csv='../data/pp_names_and_age.csv'):
    horses = []
    for file in os.listdir(directory):
        horse_lis = parse_xml_pp(os.path.join(directory, file))
        horses += horse_lis

    headers = ['reg_num', 'name', 'date_of_birth']
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

        for horse in horses:
            data = horse.regNum_name_birthdate()
            writer.writerow(data)


def update_ages_results(df_results, df_pp_ages):
    for index, row in df_results.iterrows():
        race_date, size = datetime.strptime(row['race_date'].split(' ')[0], '%Y-%m-%d').date(), row['num_of_racers']
        for i in range(size):
            horse_name = row[f'racer_{i}_name']
            if len(horse_name) > 3 and horse_name[:3] in ['dq-', 'dh-']:
                horse_name = horse_name[3:]
            try:
                birthdate = df_pp_ages.loc[df_pp_ages['name'] == horse_name, 'date_of_birth'].values[0]
                birthdate = datetime.strptime(birthdate.split('+')[0], '%Y-%m-%d').date()

                age = round((race_date - birthdate).days / 365, 2)

                df_results.at[index, f'racer_{i}_age'] = age
            except IndexError as e:
                print(horse_name, index)
                # if horse_name cannot be found, leave as is since we already have an age, just not exact with decimal

    return df_results


if __name__ == '__main__':
    data_dir_pp = '../data/2023 PPs'
    # get_regNum_name_age_pp(data_dir_pp)
    df_results = pd.read_csv('../data/output_results.csv')
    df_pp_ages = pd.read_csv('../data/pp_names_and_age.csv')

    new_df_results = update_ages_results(df_results, df_pp_ages)

    new_df_results.to_csv('../data/output_results_new.csv', index=False)
