import sys


class Racer:
    def __init__(self):
        self.name = ''
        self.equip = ''
        self.weight = 0.0
        self.post_position = ''
        self.offical_finish = ''
        self.age = 0
        self.jockey_key = 0
        self.trainer_key = 0

    @staticmethod
    def xml_headers_mapping():
        return {'ENTRY/NAME': ('name', str), 'ENTRY/EQUIP': ('equip', str), 'ENTRY/WEIGHT': ('weight', float),
                'ENTRY/POST_POS': ('post_position', str), 'ENTRY/OFFICAL_FIN': ('official_finish', str),
                'ENTRY/AGE': ('age', int), 'ENTRY/JOCKEY/KEY': ('jockey_key', int),
                'ENTRY/TRAINER/KEY': ('trainer_key', int)}

    def to_dict(self):
        return self.__dict__

    def to_csv_format(self):
        # print(','.join(str(value) for key, value in self.__dict__.items()))
        return [str(value) for key, value in self.__dict__.items()]


class Race:
    def __init__(self, date, number):
        self.date = date
        self.number = number
        self.distance = 0.0
        self.distance_unit = ''
        self.weather = ''
        self.surface = ''
        self.run_up_distance = ''
        self.track_condition = ''
        self.num_of_racers = 0
        self.racers = []

    def add_racer(self, racer: Racer):
        self.racers.append(racer)
        self.num_of_racers += 1

    @staticmethod
    def xml_headers_mapping():
        return {'DISTANCE': ('distance', float), 'WEATHER': ('weather', str), 'DIST_UNIT': ('distance_unit', str),
                'SURFACE': ('surface', str), 'RUNUPDIST': ('run_up_distance', str),
                'TRK_COND': ('track_condition', str)}

    def to_list(self):
        raw_dict = self.__dict__.copy()

        # try:
        #     raw_dict['racers'] = ','.join(racer.to_csv_format() for racer in self.racers)
        # except AttributeError as e:
        #     print(e)

        # return raw_dict

        raw_dict['racers'] = []
        for racer in self.racers:
            raw_dict['racers'] += racer.to_csv_format()

        headers = [key for key in raw_dict.keys() if key != 'racers'] + ['racers']
        values = [str(raw_dict[key]) for key in headers if key != 'racers'] + raw_dict['racers']

        return headers, values

    def update_distance(self):
        ...
