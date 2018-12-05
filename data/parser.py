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
        if not self.__parsed_data:
            return True
        else:
            return False

    def __parse(self):
        #Проверим на пустоту
        if not self.__data:
            return None

        self.__data_headers = data_headers_dict["base_headers"] + data_headers_dict[self.param_name]

        temp_data = list()
        # Разобьем каждую линию по частям
        for data_line in self.__data:
            data_line_arr = data_line.split(';')
            data_line_arr[0] = QDateTime.fromString(data_line_arr[0][:-3], "yyyy-MM-dd HH:mm:ss")

            temp_data.append(data_line_arr)

        parsed_data = sorted(temp_data, key=itemgetter(0))

        return parsed_data

    def compare_date(self, start_date, end_date):
        is_date_different = False
        if start_date != self.__parsed_data[0][0]:
            start_date = self.__parsed_data[0][0]
            is_date_different = True
        if end_date != self.__parsed_data[-1][0]:
            end_date = self.__parsed_data[-1][0]
            is_date_different = True

        return (is_date_different, start_date.toString('dd.MM.yyyy HH:mm:ss'), end_date.toString('dd.MM.yyyy HH:mm:ss'))

    def get_headers(self):
        return data_headers_dict[self.param_name]

    def get_column(self, column_header):
        column_index = self.__data_headers.index(column_header)
        column_data = {}

        if self.__parsed_data == None:
            return None

        for data_line in self.__parsed_data:
            column_data[data_line[0]] = data_line[column_index]

        return column_data

    def export_to_file(self, filename):
        with open(filename, 'w+') as file:
            file.write(self.rus_param_name + '\n')
            file.write(self.param_name + '\n')
            file.write(str(self.__station) + '\n')
            file.write(self.__detector + '\n')

            for d_line in self.__data:
                file.write(d_line + '\n')

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
            