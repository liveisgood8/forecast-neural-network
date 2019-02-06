import sys

from PyQt5.Qt import *

from modules.helper import grid_add_label_widget, show_msgbox
from modules.NNCore import NeuralNetwork, NeuralNetworkTeacher, Predictions
from modules.DataAnalyzer import DataAnalyzer
from modules.ChartView import ChartView
from form.predictions_form import PredictionsForm


class NeuralNetworkDialog(QDialog):

    def __init__(self, data_analyzer: DataAnalyzer, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Окно настройки нейронной сети')
        self.setFixedSize(900, 600)

        self.future_prediction = False
        self.data_analyzer = data_analyzer
        self.nn_teacher = None
        self.predictions = list()

        ##NN settings section
        gb_settings = QGroupBox('Настройки нейронной сети')
        gb_settings_l = QGridLayout()
        # gb_settings_l.setColumnMinimumWidth(0, int(self.width() * 0.7))
        gb_settings.setLayout(gb_settings_l)


        #Number of repeats field
        self.spin_repeat_num = QSpinBox()
        self.spin_repeat_num.setMinimum(1)
        grid_row = grid_add_label_widget(gb_settings_l, 'Количество повторов', self.spin_repeat_num, 0)

        #Number of epoch field
        self.spin_epoch_num = QSpinBox()
        self.spin_epoch_num.setMinimum(1)
        self.spin_epoch_num.setMaximum(10000)
        grid_row =grid_add_label_widget(gb_settings_l, 'Количество поколений', self.spin_epoch_num,
                                        grid_row)

        #Number of neurons field
        self.spin_neuron_num = QSpinBox()
        self.spin_neuron_num.setMinimum(1)
        grid_row = grid_add_label_widget(gb_settings_l, 'Количество нейронов в слое LSTM',
                                                    self.spin_neuron_num, grid_row)

        #Batch size field
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setMinimum(1)
        self.spin_batch_size.setMaximum(10000)
        grid_row = grid_add_label_widget(gb_settings_l, 'Размер партии (batch size)', self.spin_batch_size,
                              grid_row)

        #Size of train series
        self.spin_train_size = QSpinBox()
        self.spin_train_size.setMinimum(1)
        self.spin_train_size.setMaximum(self.data_analyzer.get_data_len())
        grid_row = grid_add_label_widget(gb_settings_l, 'Объем обучающей выборки', self.spin_train_size,
                              grid_row)

        #Number of predictions
        self.spin_predictions_num = QSpinBox()
        self.spin_predictions_num.setMinimum(1)
        grid_add_label_widget(gb_settings_l, 'Количество прогнозируемых значений', self.spin_predictions_num,
                              grid_row)

        ##NN report settings
        gb_report = QGroupBox('Отчет о работе сети')
        gb_report_l = QGridLayout()
        # gb_report_l.setColumnMinimumWidth(0, int(self.width() * 0.7))
        gb_report.setLayout(gb_report_l)

        #Current epoch
        self.r_current_epoch = QLineEdit('0')
        self.r_current_epoch.setReadOnly(True)
        grid_row = grid_add_label_widget(gb_report_l, 'Номер текущего поколения', self.r_current_epoch, 0)

        #Current repeat
        self.r_current_repeat = QLineEdit('0')
        self.r_current_repeat.setReadOnly(True)
        grid_row = grid_add_label_widget(gb_report_l, 'Номер текущего повтора', self.r_current_repeat, grid_row)


        ##NN Control buttons
        buttons_layout = QHBoxLayout()

        self.b_start = QPushButton('Начать обучение')
        self.b_start.clicked.connect(self.nn_start_education)

        self.b_stop = QPushButton('Остановить обучение')
        self.b_stop.clicked.connect(self.nn_stop_education)
        self.b_stop.setEnabled(False)

        buttons_layout.addWidget(self.b_stop)
        buttons_layout.addWidget(self.b_start)


        ##NN Report controls
        self.chart_view = ChartView(True) #No margins enabled
        self.chart_view.setFixedSize(450, 330)
        self.chart_view.x_name = "Повтор"
        self.chart_view.y_name = "Среднеквадратичное отклонение"
        self.chart_view.x_tick_num = 10
        self.chart_view.y_tick_num = 10
        self.chart_view.x_label_angle = 0
        self.chart_view.y_label_angle = 0
        self.chart_view.x_min = 1
        self.chart_view.x_max = 2
        self.chart_view.build_plot(None, None)

        self.predictions_table = QTableWidget(self)
        self.predictions_table.setColumnCount(2)
        self.predictions_table.setHorizontalHeaderLabels(['RMSE', 'Предсказанные значения'])
        self.predictions_table.horizontalHeader().setStretchLastSection(True)
        self.predictions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.predictions_table.cellDoubleClicked.connect(self.predictions_build_plot)

        #Main layout
        main_layout = QGridLayout()
        main_layout.addWidget(gb_settings, 0, 0)
        main_layout.addWidget(gb_report, 1, 0)
        main_layout.addWidget(self.chart_view, 0, 1)
        main_layout.addWidget(self.predictions_table, 1, 1)
        main_layout.addLayout(buttons_layout, 2, 0, 1, 2)
        self.setLayout(main_layout)

    def nn_start_education(self):
        self.chart_view.clean()
        self.set_cbuttons_state(False)
        self.set_spins_state(False)

        #Pick future forecast if train_size == series_size
        if self.spin_train_size.value() == self.data_analyzer.get_data_len():
            self.future_prediction = True
        else:
            self.future_prediction = False

        nn = NeuralNetwork(self.data_analyzer.data,
                           self.spin_train_size.value(),
                           self.spin_predictions_num.value(),
                           self.spin_repeat_num.value(),
                           self.spin_epoch_num.value(),
                           self.spin_batch_size.value(),
                           self.spin_neuron_num.value())

        self.nn_teacher = NeuralNetworkTeacher(nn, self)
        self.nn_teacher.signal_epoch.connect(self.increment_epoch)
        self.nn_teacher.signal_repeat.connect(self.increment_repeats)
        self.nn_teacher.signal_complete.connect(self.teaching_complete)
        self.nn_teacher.start()

    def nn_stop_education(self):
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        if self.nn_teacher:
            self.nn_teacher.tterminate()
            if self.nn_teacher.wait(5000):
                QGuiApplication.setOverrideCursor(Qt.ArrowCursor)
                self.set_cbuttons_state(True)
                self.set_spins_state(True)
            else:
                QGuiApplication.setOverrideCursor(Qt.ArrowCursor)
                show_msgbox('Ошибка при остановке обучения, перезапустите программу!', True)

    def increment_epoch(self, count):
        self.r_current_epoch.setText(str(count + 1))

    def increment_repeats(self, count, predictions: Predictions):
        self.r_current_repeat.setText(str(count + 1))

        if not self.future_prediction:
            self.chart_view.series_append(count + 1, predictions.rmse, 1, self.get_min_rmse(),
                                          True, predictions.rmse > self.get_max_rmse())
        self.predictions.append(predictions)

        #Add result in predictio_table
        row_num = self.predictions_table.rowCount()
        predictions_str = ', '.join(str(round(e, 4)) for e in predictions.values)
        self.predictions_table.insertRow(row_num)

        r_item = QTableWidgetItem(str(round(predictions.rmse, 4))
                                  if predictions.rmse != NeuralNetwork.RMSE_ABORTED_VALUE
                                  else 'NaN')
        p_item = QTableWidgetItem(predictions_str)
        self.predictions_table.setItem(row_num, 0, r_item)
        self.predictions_table.setItem(row_num, 1, p_item)

    def predictions_build_plot(self, row, column):
        print(self.predictions[row].values)
        predictions_form = PredictionsForm(self.data_analyzer, self)
        predictions_form.exec()

    def teaching_complete(self):
        self.set_cbuttons_state(True)
        self.set_spins_state(True)

    def set_spins_state(self, state):
        self.spin_batch_size.setEnabled(state)
        self.spin_neuron_num.setEnabled(state)
        self.spin_epoch_num.setEnabled(state)
        self.spin_repeat_num.setEnabled(state)

    def clean_report(self):
        self.r_current_epoch.setText('0')
        self.r_current_repeat.setText('0')

    def set_cbuttons_state(self, state):
        self.b_start.setEnabled(state)
        self.b_stop.setEnabled(not state)

    def get_max_rmse(self):
        max_rmse = 0
        for elem in self.predictions:
            if elem.rmse > max_rmse:
                max_rmse = elem.rmse

        return max_rmse

    def get_min_rmse(self):
        if len(self.predictions) == 0:
            return 0

        min_rmse = sys.float_info.max
        for elem in self.predictions:
            if elem.rmse < min_rmse:
                min_rmse = elem.rmse

        return min_rmse







