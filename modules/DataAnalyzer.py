import numpy as np
import scipy.stats as scistats
import scipy.fftpack as scifft

class DataAnalyzer():
    def set_data(self, data):
        self.data = data

        self.data_values = list()
        for item in data.values():
            self.data_values.append(float(item))

    #Нормализация данных
    def convert_data(self):
        converted_data = {}

        for elem in self.data.keys():
            converted_data[elem.toMSecsSinceEpoch()] = float(self.data.get(elem))

        return converted_data

    def fft(self):
        # Количество измерений
        N = len(self.data_values)

        # Частота дискретизации
        Fs = 1.0 / self.get_time_delta_of_measure()

        yf = scifft.fft(self.data_values)
        xf = np.linspace(0.0, 1.0 / (2.0 * Fs), N // 2)

        yff = 2.0 / N * np.abs(yf[0:N // 2])

        return xf[1:], yff[1:]

    def get_time_delta_of_measure(self):
        keys = list(self.data.keys())
        return keys[0].secsTo(keys[1])

    def nextpow2(i):
        n = 1
        while n < i: n *= 2
        return n

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
