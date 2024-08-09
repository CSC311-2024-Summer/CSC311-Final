import xml.etree.ElementTree as ET
import csv
import os
from objects import Racer, Race, Horse, PastPerformance
from datetime import datetime, timedelta


def parse_element_helper(element, parent_path=""):
    results = []
    for child in element:
        child_path = f"{parent_path}/{child.tag}" if parent_path else child.tag
        if child.text and child.text.strip():
            results.append(f"{child_path}: {child.text.strip()}")
        results.extend(parse_element_helper(child, child_path))
    return results


def get_nested_data_as_string(element):
    parsed_data = parse_element_helper(element)
    return ", ".join(parsed_data)


# result charts data:
def update_racer_result_charts(element, cur_racer, prefix=''):
    for child in element:
        column_name = f"{prefix}{child.tag}" if prefix else child.tag
        if column_name in Racer.xml_headers_mapping():
            # print(column_name, child.text)
            attr, attr_type = Racer.xml_headers_mapping()[column_name]
            converted_val = attr_type(child.text)
            setattr(cur_racer, attr, converted_val)

            # seperate case (not automated)
            if column_name == 'ENTRY/LAST_PP':
                cur_racer.last_pp = get_nested_data_as_string(child)

        update_racer_result_charts(child, cur_racer, f"{column_name}/")


def parse_xml_result_charts(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    file_date = root.get('RACE_DATE')

    races = []
    for race in root.findall('RACE'):
        cur_race = Race(date=datetime.strptime(file_date, '%Y-%m-%d'), number=race.get('NUMBER'))
        for column, (attr, attr_type) in Race.xml_headers_mapping().items():
            converted_val = attr_type(race.find(column).text.strip()) if race.find(column) is not None else ''
            setattr(cur_race, attr, converted_val)

        for racer in race.findall('ENTRY'):
            cur_racer = Racer()
            update_racer_result_charts(racer, cur_racer, 'ENTRY/')
            cur_race.add_racer(cur_racer)

        races.append(cur_race)

    return races


# pp data:
def update_horse_pp(element, cur_pp, prefix=''):
    for child in element:
        column_name = f"{prefix}{child.tag}" if prefix else child.tag
        if column_name in PastPerformance.xml_headers_mapping():
            # print(column_name, child.text)
            attr, attr_type = PastPerformance.xml_headers_mapping()[column_name]
            try:
                converted_val = attr_type(child.text)
                setattr(cur_pp, attr, converted_val)
            except TypeError as e:
                print(child.text, child.tag)

        update_horse_pp(child, cur_pp, f"{column_name}/")


def parse_xml_pp(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    horse_info = []
    for race in root.findall('Race'):
        for starter in race.findall('Starters'):
            horse = starter.find('Horse')
            cur_horse = Horse()
            for column, (attr, attr_type) in Horse.xml_headers_mapping().items():
                # if column == 'BreedType/Value':
                #     cur_horse.breed_type = horse.find('BreedType').find('Value').text.strip()
                #     continue
                try:
                    converted_val = attr_type(horse.find(column).text.strip()) if horse.find(column) is not None else ''
                    setattr(cur_horse, attr, converted_val)
                except ValueError or TypeError as e:
                    print(e, horse.find(column).tag, column)

            for past_performance in starter.findall('PastPerformance'):
                cur_pp = PastPerformance()
                update_horse_pp(past_performance, cur_pp, '/')

                # handling Fractions because it has special cases
                for fractions in root.findall('Fractions'):
                    fraction = fractions.find('Fraction').text
                    # cur_pp.finish_time += int(fractions.find('Time').text)
                    if fraction == 'W':
                        cur_pp.finish_time = fractions.find('Time').text
                        break

                cur_pp.update()

                cur_horse.add_pp(cur_pp)

            horse_info.append(cur_horse)

    return horse_info


def write_to_csv_pp(horse_files, output_csv):
    headers = ['reg_num', 'sex', 'age'] + list(PastPerformance().to_csv_format().keys())
    headers.remove('wind_dir')
    headers.remove('wind_speed')
    headers.remove('distance_unit')
    headers.remove('race_number')

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        # for horses in horse_files:
        #     for horse in horses:
        #         data = horse.to_csv_format()
        #         writer.writerows(data)
        for horse in horse_files:
            data = horse.to_csv_format()
            # print('data', data)
            writer.writerows(data)


def write_to_csv_results(races, output_csv):
    # create headers based on the max number of racers within a race
    max_racers = max(race.num_of_racers for race in races)
    headers = races[0].to_list()[0]
    headers += [f'racer_{i}_{key}' for i in range(max_racers) for key in Racer().to_dict().keys()]

    # Write to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for race in races:
            if isinstance(race.racers, str):
                raise AttributeError
            writer.writerow(race.to_list()[1])


def parse_all_files_results(directory, output_csv):
    races = []
    # nums = {}
    for file in os.listdir(directory):
        race_lis = parse_xml_result_charts(os.path.join(directory, file))
        # for race in race_lis:
        #     if isinstance(race.racers, str):
        #         raise AttributeError
        #     if race.num_of_racers not in nums:
        #         nums[race.num_of_racers] = 1
        #     else:
        #         nums[race.num_of_racers] += 1
        races += race_lis

    write_to_csv_results(races, output_csv)
    # print(nums)


def parse_all_files_pp(directory, output_csv):
    horses = []
    for file in os.listdir(directory):
        horse_lis = parse_xml_pp(os.path.join(directory, file))
        horses += horse_lis

    write_to_csv_pp(horses, output_csv)


def aggregate_previous_year_averages(df, row, column='official_finish', range=365 * 3):
    """
    :param df:
    :param column: precondtion - df[column] must be summable (float or int)
    :return:
    """
    filtered_df = df[(df['reg_num'] == row['reg_num'])
                     & (df['race_date'] >= (row['race_date'] - timedelta(days=range)))
                     & (df['race_date'] <= row['race_date'])]

    if filtered_df.empty:
        return 0

    return filtered_df[column].mean()


if __name__ == '__main__':
    data_dir = '../data/2023 Result Charts'
    output_csv = './data/output.csv'

    # TESTING
    # data_file = './data/2023 Result Charts/wo20230422tch.xml'
    # races = parse_xml(data_file)
    # for race in races:
    #     print(race.to_dict())
    #
    # print('============')
    #
    # print(races[0].racers[0].to_dict())

    # parse_all_files_results(data_dir, output_csv)

    data_dir_pp = '../data/2023 PPs'
    output_csv_pp = './data/pp_output_new.csv'
    # parse_all_files_pp(data_dir_pp, output_csv_pp)

    import pandas as pd

    # df = pd.read_csv(output_csv_pp)
    # df['race_date'] = pd.to_datetime(df['race_date'], format='%Y-%m-%d+%H:%M')
    # earliest_date = df['race_date'].min()
    # print('earliest date is: ', earliest_date)
    #
    # threshold_date = pd.to_datetime('2020-01-01')
    # df_clean = df[df['race_date'] > threshold_date]
    #
    # df_clean['scratched'] = df_clean['finish_time'] == 0
    # df_clean['avg_official_finish'] = df_clean.apply(lambda row: aggregate_previous_year_averages(df_clean, row, 'official_finish'), axis=1)
    # df_clean = df_clean.drop(columns=['official_finish', 'time_of_horse'])
    # df_clean.to_csv('./data/cleaned_output_pp_new.csv', index=False)

    # round avg official finish time to nearest 0.5
    df = pd.read_csv('../data/cleaned_output_pp_new.csv')
    df['avg_official_finish'] = df['avg_official_finish'].apply(lambda x: round(x))
    df['equipment'] = df['equipment'].str.extract(r'Value: (.*)')
    df['equipment'] = df['equipment'].fillna('N/A')
    df['age'] = df['age'].round(2)
    df.to_csv('../data/output_pp.csv', index=False)
