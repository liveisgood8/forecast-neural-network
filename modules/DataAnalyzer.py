import numpy as np
import pandas

import scipy.stats as scistats
import scipy.fftpack as scifft

from modules.parser import DataParser

class DataAnalyzer(DataParser):
    def __init__(self, rus_param_name, param_name, station, detector, parsed_data=None):
        super().__init__(rus_param_name, param_name, station, detector, parsed_data)
        self.data = None
        self.data_values = None

    def set_data(self, data: pandas.DataFrame):
        self.data = data
        self.data_values = data.iloc[:, 1]

    #Преобразование данных для построения графика
    @staticmethod
    def convert_data(data):
        x = list()
        y = list()

        for index, row in data.iterrows():
            x.append(row.iloc[0].toMSecsSinceEpoch())
            y.append(float(row.iloc[1]))

        return x, y

    #Преобразует лист предсказаний в time series, начальное время определяется по объему обуч. выборки
    def predictions_to_timeseries(self, predictions, train_size):
        time_series_list = list()

        time_delta = self.get_time_delta_of_measure()
        data_times = self.data.iloc[:, 0]

        start_time = data_times.iloc[-1]
        if train_size < self.get_data_len():
            start_time = data_times.iloc[train_size - 1]

        for pred in predictions:
            start_time = start_time.addSecs(time_delta)
            time_series_list.append([start_time, pred])

        print(time_series_list)
        time_series_frame = pandas.DataFrame(time_series_list)
        return time_series_frame

    def fft(self):
        # TEST SIGNAL
        # Fs = 1000
        # T = 1 / Fs
        # N = 100
        # t = np.arange(0, N) * T
        #
        # S = 2 * np.sin(2 * np.pi * 50 * t) + 5 * np.sin(2 * np.pi * 70 * t)
        #
        # yf = scifft.fft(S)
        # yff = 2.0 / N * np.abs(yf[0:N // 2])
        # xf = Fs * np.arange(0, N//2) / N

        #Info: https://www.mathworks.com/help/matlab/ref/fft.html

        N = len(self.data_values) #Num of measures
        Fs = 1 / self.get_time_delta_of_measure() #Sampling frequency - num of measures in 1 second

        yf = scifft.fft(self.data_values)
        yff = 2.0 / N * np.abs(yf[0:N // 2])
        yff_phase = np.angle(yf, deg=True)
        xf = Fs * np.arange(0, N//2) / N

        return xf[1:], yff[1:], yff_phase[1:], yf[1:].round(5)

    def get_time_delta_of_measure(self):
        return self.data.iloc[:,0][0].secsTo(self.data.iloc[:,0][1])

    def get_data_len(self):
        return len(self.data_values)

    def min(self):
        return np.amin(self.data_values)

    def max(self):
        return np.amax(self.data_values)

    def mean(self):
        return np.mean(self.data_values)

    def std(self):
        return np.std(self.data_values)

    def median(self):
        return np.median(self.data_values)

    def mode(self):
        return scistats.mode(self.data_values)[0][0]

    def min_max_delta(self):
        return np.amax(self.data_values) - np.amin(self.data_values)
