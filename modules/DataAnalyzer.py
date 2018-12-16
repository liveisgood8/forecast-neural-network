import numpy as np
import pandas

import scipy.stats as scistats
import scipy.fftpack as scifft

class DataAnalyzer():
    def set_data(self, data: pandas.DataFrame):
        self.data = data
        self.data_values = data.iloc[:,1]

    #Преобразование данных для построения графика
    def convert_data(self):
        x = list()
        y = list()

        for index, row in self.data.iterrows():
            x.append(row.iloc[0].toMSecsSinceEpoch())
            y.append(float(row.iloc[1]))

        return x, y

    def fft(self):
        # TEST SIGNAL
        # Fs = 1000
        # T = 1 / Fs
        # N = 100
        # t = np.arange(0, N) * T
        #
        # S = 2 * np.sin(2 * np.pi * 50 * t)
        #
        # yf = scifft.fft(S)
        # yff = 2.0 / N * np.abs(yf[0:N // 2])
        # xf = Fs * np.arange(0, N//2) / N

        #Info: https://www.mathworks.com/help/matlab/ref/fft.html

        N = len(self.data_values) #Num of measures
        Fs = 1 / self.get_time_delta_of_measure() #Sampling frequency - num of measures in 1 second

        yf = scifft.fft(self.data_values)
        yff = 2.0 / N * np.abs(yf[0:N // 2])
        xf = Fs * np.arange(0, N//2) / N

        return xf[1:], yff[1:], yf[1:]

    def get_time_delta_of_measure(self):
        return self.data.iloc[:,0][0].secsTo(self.data.iloc[:,0][1]);

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
