import pandas
import os
from io import StringIO

from PyQt5.QtCore import QDateTime
from operator import itemgetter

from data.dictionaries import data_headers_dict


class DataParser:
    def __init__(self, rus_param_name=None, param_name=None):
        self.param_name = param_name
        self.rus_param_name = rus_param_name

    def set_data(self, data):
        self.__data = data
        self.__parsed_data = self.__parse()

    def set_station(self, station):
        self.__station = station

    def set_detector(self, detector):
        self.__detector = detector

    def is_data_empty(self):
        if self.__parsed_data.empty:
            return True
        else:
            return False

    def __parse(self):
        #Проверим на пустоту
        if not self.__data:
            return None

        data_stream = StringIO(self.__data)
        parsed_data = pandas.read_csv(data_stream, names=data_headers_dict["base_headers"] + data_headers_dict[self.param_name],
                              sep=';', index_col=False)

        parsed_data['Время измерения'] = self.convert_str_to_qdatetime(parsed_data['Время измерения'])
        parsed_data = parsed_data.sort_values('Время измерения')

        return parsed_data

    def compare_date(self, start_date, end_date):
        is_date_different = False
        if start_date != self.__parsed_data.ix[:,0].iloc[0]:
            start_date = self.__parsed_data.ix[:,0].iloc[0]
            is_date_different = True
        if end_date != self.__parsed_data.ix[:,0].iloc[-1]:
            end_date = self.__parsed_data.ix[:,0].iloc[-1]
            is_date_different = True

        return (is_date_different, start_date.toString('dd.MM.yyyy HH:mm:ss'), end_date.toString('dd.MM.yyyy HH:mm:ss'))

    def convert_str_to_qdatetime(self, column_values):
        datetime_list = list()
        for elem in column_values:
            datetime_list.append(QDateTime.fromString(elem[:-3], "yyyy-MM-dd HH:mm:ss"))
        return datetime_list

    def convert_qdatetime_to_str(self, column_values):
        datetime_list = list()
        for elem in column_values:
            datetime_list.append(elem.toString("yyyy-MM-dd HH:mm:ss"))
        return datetime_list

    def get_headers(self):
        return data_headers_dict[self.param_name]

    def get_column(self, column_header):
        return self.__parsed_data[['Время измерения', column_header]]

    def export(self, filename):
        self.__parsed_data['Время измерения'] = self.convert_qdatetime_to_str(self.__parsed_data['Время измерения'])

        extension = os.path.splitext(filename)[1]
        if extension == '.xlsx':
            self.__parsed_data.to_excel(filename)
        elif extension == '.csv':
            self.__parsed_data.to_csv(filename, sep=';')

        self.__parsed_data['Время измерения'] = self.convert_str_to_qdatetime(self.__parsed_data['Время измерения'])


    def import_from_file(self, filename):
        with open(filename, 'r+') as file:
            content = file.readlines()
            data_lines = [x.strip() for x in content]

            self.rus_param_name = data_lines[0]
            self.param_name = data_lines[1]
            self.set_station(int(data_lines[2]))
            self.set_detector(data_lines[3])

            self.__data = data_lines[4:]
            self.__parsed_data = self.__parse()
            