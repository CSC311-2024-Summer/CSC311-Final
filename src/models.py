import sys
from datetime import datetime


class Racer:
    def __init__(self):
        self.name = ''
        self.equip = ''
        self.weight = 0.0
        self.post_position = ''
        self.official_finish = ''
        self.age = 0
        self.jockey_key = 0
        self.trainer_key = 0
        # self.breed_type = '' # always TB
        self.last_pp = ''
        self.meds = ''
        self.dollar_odds = 0.0
        self.comments = ''

    @staticmethod
    def xml_headers_mapping():
        return {
            'ENTRY/NAME': ('name', str), 'ENTRY/EQUIP': ('equip', str), 'ENTRY/WEIGHT': ('weight', float),
            'ENTRY/POST_POS': ('post_position', str),
            'ENTRY/OFFICIAL_FIN': ('official_finish', str),
            'ENTRY/AGE': ('age', int), 'ENTRY/JOCKEY/KEY': ('jockey_key', int),
            'ENTRY/TRAINER/KEY': ('trainer_key', int), 'ENTRY/LAST_PP': ('last_pp', str),
            'ENTRY/MEDS': ('meds', str), 'ENTRY/DOLLAR_ODDS': ('dollar_odds', float),
            # 'ENTRY/BREED': ('breed_type', str), # always TB
            'ENTRY/COMMENT': ('comments', str)
        }

    def to_dict(self):
        return self.__dict__.copy()

    def to_csv_format(self):
        # print(','.join(str(value) for key, value in self.__dict__.items()))
        # return [str(key) + ': ' + str(value) for key, value in self.__dict__.items()]
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

        headers = [key for key in raw_dict.keys() if key != 'racers']
        headers.remove('distance_unit')
        values = [str(raw_dict[key]) for key in headers if key != 'racers'] + raw_dict['racers']
        values[headers.index('distance')] = self.convert_distance_to_meters()

        return headers, values

    def to_csv_format(self):
        raw_dict = self.__dict__.copy()
        del raw_dict['racers']

        for i, racer in enumerate(self.racers, start=1):
            for key, value in racer.to_dict():
                raw_dict[f'racer_{i}_{key}'] = value

        return raw_dict

    def convert_distance_to_meters(self):
        if self.distance_unit == 'F':
            return round(self.distance / 25) * 201.168 * 25 / 100
        if self.distance_unit == 'Y':
            return self.distance * 0.9144
        if self.distance_unit == 'M':
            return (self.distance / 2) * 201.168


class PastPerformance:
    def __init__(self):
        self.num_of_starters = 0
        self.race_date = ''
        self.run_up_distance = 0
        self.race_number = 0
        self.race_comments = ''
        self.track_condition = ''
        self.wind_dir = ''
        self.wind_speed = 0.0
        self.temperature = ''
        self.distance = 0.0
        self.distance_unit = ''
        self.weather = ''
        self.post_position = ''
        self.official_finish = ''
        self.time_of_horse = ''
        self.equipment = ''
        self.jockey = 0
        self.weight_carried = 0
        self.meds = ''
        self.dollar_odds = 0.0
        self.finish_time = 0
        self.owner = ''
        self.trainer = ''

    @staticmethod
    def xml_headers_mapping():
        d_temp = {
            '/NumberOfStarters': ('num_of_starters', int),
            '/RaceDate': ('race_date', str),
            '/RunUpDistance': ('run_up_distance', str),
            '/RaceNumber': ('race_number', int),
            '/Start/LongComment': ('race_comments', str),
            # RaceComment under PastPerformance is always blank
            '/TrackCondition/Value': ('track_condition', str),
            '/WindDirection': ('wind_dir', str),
            '/WindSpeed': ('wind_speed', float),
            '/Temperature': ('temperature', str),
            '/Distance/DistanceId': ('distance', float),
            '/Distance/DistanceUnit/Value': ('distance_unit', str),
            '/Weather': ('weather', str),
            '/Start/PostPosition': ('post_position', str),
            '/Start/OfficialFinish': ('official_finish', float),
            '/Start/TimeOfHorse': ('time_of_horse', str),
            '/Start/Equipment/Value': ('equipment', str),
            '/Start/Jockey/ExternalPartyId': ('jockey', int),
            '/Start/WeightCarried': ('weight_carried', float),
            '/Start/Medication/Description': ('meds', str),  # need inner parsing
            '/Start/Odds': ('dollar_odds', float),
            '/Fractions/Time': ('finish_time', str),  # only if Fractions/Fraction == 'W'
            '/Start/Owner/ExternalPartyId': ('owner', str),
            '/Start/Trainer/ExternalPartyId': ('trainer', str)
        }

        # new_d = {}
        #
        # for key in d_temp:
        #     new_d['Starter/PastPerformance' + key] = d_temp[key]
        #
        # return new_d
        return d_temp

    def to_list(self):
        return [str(key) + ': ' + str(value) for key, value in self.__dict__.items()]

    def to_csv_format(self):
        d = self.__dict__.copy()
        return d

    def convert_distance(self):
        if self.distance_unit == 'F':
            return round(self.distance / 25) * 201.168 * 25 / 100
        if self.distance_unit == 'Y':
            return self.distance * 0.9144
        if self.distance_unit == 'M':
            return (self.distance / 2) * 201.168

    def update(self):
        self.dollar_odds = self.dollar_odds / 100
        self.finish_time = int(self.finish_time) / 1000


class Horse:
    """
    Nessecary features and information about each horse (ie. Name, date of birth) must be extracted from
        <Starter/Horse>, however past performance data (ie. finsih time) must be extracted from
        <Starter/PastPerformances/Start> so this class requires a hierarchal storing structure
    """
    past_performances: list[PastPerformance]

    def __init__(self):
        self.reg_num = 0
        self.name = ''
        # self.breed_type = ''
        self.date_of_birth = ''  # year of birth changed to foaling date
        self.sex = ''
        self.past_performances = []
        # self.name = ''
        # self.equip = ''
        # self.weight = 0.0
        # self.post_position = ''
        # self.age = 0
        # self.jockey_key = 0
        # self.trainer_key = 0
        # self.breed_type = ''
        # self.last_pp = ''
        # self.comments = ''
        # self.past_performances = []

    @staticmethod
    def xml_headers_mapping():
        return {
            'RegistrationNumber': ('reg_num', str),
            'HorseName': ('name', str),
            # 'BreedType/Value': ('breed_type', str),
            'FoalingDate': ('date_of_birth', str),
            'Sex/Value': ('sex', str)
        }
        # {
        #     'Starters/Horse/HorseName': ('name', str), 'Starters/Equipment': ('equip', str),
        #     'Starters/WeightCarried': ('weight', float), 'Starters/PostPosition': ('post_position', str),
        #     'Starters/Horse/Year': ('age', int), 'Starters/Jockey/ExternalPartyId': ('jokey_key', int),
        #     'Starters/Trainer/ExternalPartyId': ('trainer_key', int), 'Starters/PastPerformance': ('last_pp', str),
        #     'Starters/Medication': ('meds', str), 'Starters/Odds': ('odds', float),
        #     'Starters/Horse/BreedType/Value': ('breed_type', str),
        #     'Starters/CommentText': ('comments', str)
        # }

    def to_list(self):
        ...

    def find_age(self, race_date: datetime):
        age = race_date - datetime.strptime(self.date_of_birth, '%Y-%m-%d+%H:%M')
        return age.days / 365

    def to_csv_format(self) -> list[dict]:
        """
        :return: This function returns a list with all the past races this horse has competed in
        """
        all_pps = []
        csv_dict = {
            'reg_num': self.reg_num,
            'sex': self.sex
        }

        for race in self.past_performances:
            cur_csv_dict = csv_dict.copy()
            cur_csv_dict['age'] = self.find_age(datetime.strptime(race.race_date, '%Y-%m-%d+%H:%M'))
            cur_csv_dict.update(race.to_csv_format())
            cur_csv_dict['distance'] = race.convert_distance()
            del cur_csv_dict['wind_dir']
            del cur_csv_dict['wind_speed']
            del cur_csv_dict['distance_unit']
            del cur_csv_dict['race_number']
            all_pps.append(cur_csv_dict)
        return all_pps

    def add_pp(self, new_pp: PastPerformance):
        self.past_performances.append(new_pp)
