from PyQt5.QtWidgets import *

from modules.ChartView import ChartView
from modules.DataAnalyzer import DataAnalyzer


class PredictionsForm(QDialog):
    def __init__(self, data_analyzer: DataAnalyzer, train_size, predictions, parent=None):
        super().__init__(parent)

        self.data_analyzer = data_analyzer

        self.setWindowTitle('Графическое представление прогноза')
        self.setFixedSize(1000, 800)

        self.chart_view = ChartView()
        self.chart_view.x_time_scaled = True
        self.chart_view.x_name = 'Время'
        self.chart_view.y_name = self.data_analyzer.selected_column

        ##Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.chart_view)

        self.setLayout(main_layout)

        ##Построение графика
        #Получим необходимую колонку
        data = self.data_analyzer.get_selected_column()
        self.data_analyzer.set_data(data)
        x, y = DataAnalyzer.convert_data(self.data_analyzer.data)

        #Строим базовый график
        self.chart_view.build_plot((x, y), self.data_analyzer.rus_param_name)
        # pred_df = self.data_analyzer.predictions_to_timeseries(predictions[0], train_size)
        # pred_time_series = self.data_analyzer.convert_data(pred_df)
        # self.chart_view.build_multiple_plot((x, y), pred_time_series, self.data_analyzer.rus_param_name)

        #Конвертим предсказания и добавляем к графику
        for pred in predictions:
            pred_df = self.data_analyzer.predictions_to_timeseries(pred, train_size)
            pred_time_series = self.data_analyzer.convert_data(pred_df)
            self.chart_view.add_series(pred_time_series)
            train_size += 1


