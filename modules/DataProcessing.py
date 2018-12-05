from modules.DataAnalyzer import DataAnalyzer

class DataProcessing(DataAnalyzer):
    def normalize_data(self):
        abs_max = max(super().min(), super().max(), key=abs) #Макс. модуль ряда

        normalized_data = {}
        for elem in self.data.keys():
            normalized_data[elem] = float(self.data.get(elem)) / abs_max

        print(normalized_data)

        return normalized_data


    #Преобразование данных для построения графика
    def convert_data(self):
        x = list()
        y = list()

        for elem in self.data.keys():
            x.append(elem.toMSecsSinceEpoch())
            y.append(float(self.data.get(elem)))

        return x, y
