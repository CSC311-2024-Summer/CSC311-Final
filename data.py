import xml.etree.ElementTree as ET
import csv
import os
from models import Racer, Race
from datetime import datetime


def update_racer(element, cur_racer, prefix=''):
    for child in element:
        column_name = f"{prefix}{child.tag}" if prefix else child.tag
        if column_name in Racer.xml_headers_mapping():
            # print(column_name, child.text)
            attr, attr_type = Racer.xml_headers_mapping()[column_name]
            converted_val = attr_type(child.text)
            setattr(cur_racer, attr, converted_val)
        update_racer(child, cur_racer, f"{column_name}/")


def parse_xml(file_path):
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
            update_racer(racer, cur_racer, 'ENTRY/')
            if isinstance(cur_racer, str):
                raise AttributeError
            cur_race.add_racer(cur_racer)
        if isinstance(cur_race.racers, str):
            raise AttributeError
        races.append(cur_race)

    return races


def write_to_csv(races, output_csv):
    # Collect all headers
    headers = races[0].to_list()[0]

    # Write to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for race in races:
            if isinstance(race.racers, str):
                raise AttributeError
            writer.writerow(race.to_list()[1])


def parse_all_files(directory, output_csv):
    races = []
    for file in os.listdir(directory):
        race_lis = parse_xml(os.path.join(directory, file))
        for race in race_lis:
            if isinstance(race.racers, str):
                raise AttributeError
        races += race_lis

    write_to_csv(races, output_csv)


if __name__ == '__main__':
    data_dir = './data/2023 Result Charts'
    output_csv = './data/test.csv'

    # TESTING
    # data_file = './data/2023 Result Charts/wo20230422tch.xml'
    # races = parse_xml(data_file)
    # for race in races:
    #     print(race.to_dict())
    #
    # print('============')
    #
    # print(races[0].racers[0].to_dict())

    parse_all_files(data_dir, output_csv)
