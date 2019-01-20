from multiprocessing import Pool

from PyQt5.QtWidgets import *

from modules.helper import grid_add_label_widget, show_msgbox
from modules.NNCore import NeuralNetwork, NeuralNetworkTeacher, Predictions
from modules.DataAnalyzer import DataAnalyzer


class NeuralNetworkDialog(QDialog):

    def __init__(self, data_analyzer : DataAnalyzer, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Окно настройки нейронной сети')
        self.setFixedSize(500, 350)

        self.data_analyzer = data_analyzer
        self.nn_teacher = None

        ##NN settings section
        gb_settings = QGroupBox('Настройки нейронной сети')
        gb_settings_l = QGridLayout()
        gb_settings_l.setColumnMinimumWidth(0, int(self.width() * 0.7))
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
        grid_add_label_widget(gb_settings_l, 'Размер партии (batch size)', self.spin_batch_size,
                              grid_row)


        ##NN report settings
        gb_report = QGroupBox('Отчет о работе сети')
        gb_report_l = QGridLayout()
        gb_report_l.setColumnMinimumWidth(0, int(self.width() * 0.7))
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


        #Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(gb_settings)
        main_layout.addWidget(gb_report)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def nn_start_education(self):
        self.set_cbuttons_state(False)
        self.set_spins_state(False)

        nn = NeuralNetwork(self.data_analyzer.data,
                           self.spin_repeat_num.value(),
                           self.spin_epoch_num.value(),
                           self.spin_batch_size.value(),
                           self.spin_neuron_num.value())

        # TODO Increment
        # lstm_model = nn.fit_lstm(
        #     lambda context=self: context.r_current_epoch.setText(str(int(context.r_current_epoch.toPlainText()) + 1)))

        self.nn_teacher = NeuralNetworkTeacher(nn, self)
        self.nn_teacher.signal_epoch.connect(self.increment_epoch)
        self.nn_teacher.signal_repeat.connect(self.increment_repeats)
        self.nn_teacher.signal_complete.connect(self.teaching_complete)
        self.nn_teacher.start()

    def nn_stop_education(self):
        if self.nn_teacher:
            self.nn_teacher.terminate()
            if self.nn_teacher.wait(2000):
                self.set_cbuttons_state(True)
                self.set_spins_state(True)
            else:
                show_msgbox('Ошибка при остановке обучения, перезапустите программу!', True)

    def increment_epoch(self, count):
        self.r_current_epoch.setText(str(count + 1))

    def increment_repeats(self, count, predictions : Predictions):
        self.r_current_repeat.setText(str(count + 1))
        print(predictions.rmse, predictions.values)

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







