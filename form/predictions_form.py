from PyQt5.QtWidgets import *
from PyQt5.Qt import *

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
        # self.chart_view.chart().legend().setBackgroundVisible(True)
        # self.chart_view.chart().legend().setPen(QPen(QColor(192, 192, 192, 192)))
        # self.chart_view.chart().legend().setAlignment(Qt.AlignBottom)

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
        self.chart_view.build_plot((x, y), self.data_analyzer.rus_param_name, True, 'Original')

        #Конвертим предсказания и добавляем к графику
        for idx, pred in enumerate(predictions):
            pred_df = self.data_analyzer.predictions_to_timeseries(pred, train_size)
            pred_time_series = self.data_analyzer.convert_data(pred_df)
            self.chart_view.add_series(pred_time_series, 'Prediction ' + str(idx))
            train_size += 1


