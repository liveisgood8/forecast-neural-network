from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QGroupBox, QLabel, QSpinBox, \
    QTextEdit

from modules.helper import grid_add_label_widget

from modules.NNCore import NeuralNetwork
from modules.DataAnalyzer import DataAnalyzer

class NeuralNetworkDialog(QDialog):

    def __init__(self, data_analyzer : DataAnalyzer, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Окно настройки нейронной сети')
        self.setMinimumSize(600, 900)

        self.data_analyzer = data_analyzer

        ##NN settings section
        gb_settings = QGroupBox('Настройки нейронной сети')
        gb_settings_l = QGridLayout()
        gb_settings.setLayout(gb_settings_l)


        #Number of repeats field
        self.spin_repeat_num = QSpinBox()
        self.spin_repeat_num.setMinimum(1)
        grid_row = grid_add_label_widget(gb_settings_l, 'Количество повторов', self.spin_repeat_num, 0)

        #Number of epoch field
        self.spin_epoch_num = QSpinBox()
        self.spin_epoch_num.setMinimum(1)
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
        grid_add_label_widget(gb_settings_l, 'Размер партии (batch size)', self.spin_batch_size,
                              grid_row)


        ##NN report settings
        gb_report = QGroupBox('Отчет о работе сети')
        gb_report_l = QGridLayout()
        gb_report.setLayout(gb_report_l)

        #Current repeat
        self.r_current_repeat = QTextEdit('0')
        self.r_current_repeat.setEnabled(False)
        grid_row = grid_add_label_widget(gb_report_l, 'Номер текущего повтора', self.r_current_repeat, 0)

        #Current epoch
        self.r_current_epoch = QTextEdit('0')
        self.r_current_epoch.setEnabled(False)
        grid_row = grid_add_label_widget(gb_report_l, 'Номер текущего поколения', self.r_current_epoch, grid_row)


        #Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(gb_settings)
        main_layout.addWidget(gb_report)
        self.setLayout(main_layout)

        nn = NeuralNetwork(self.data_analyzer.data, 1, 1000, 1, 6)

        #TODO Increment
        lstm_model = nn.fit_lstm(
            lambda context=self: context.r_current_epoch.setText(str(int(context.r_current_epoch.toPlainText()) + 1)))

    def test(self):
        self.r_current_epoch.setText('10')





