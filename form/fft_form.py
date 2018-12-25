import sys
import numpy as np

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QGroupBox, QLabel, QRadioButton

from modules.ChartView import ChartView


class FftDialog(QDialog):

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Преобразование фурье')
        self.setMinimumSize(1200, 800)
        self.ampl_data = [data[0], data[1]]
        self.phase_data = [data[0], data[2]]

        #Виджет с графиком
        self.chart_view = ChartView()
        self.chart_view.setMinimumSize(1000, 800)


        #Виджет для коэффициентов
        fft_coef_list = QListWidget()
        fft_coef_label = QLabel('Коэффициенты фурье:')
        for elem in data[3]:
            str_elem = str(elem).replace('(', '')
            str_elem = str_elem.replace(')', '')
            fft_coef_list.addItem(str_elem)

        self.ampl_radio = QRadioButton()
        self.ampl_radio.setText('Амлитудный спектр')
        self.ampl_radio.toggled.connect(self.ampl_radio_toggled)

        self.phase_radio = QRadioButton()
        self.phase_radio.setText('Фазовый спектр')
        self.phase_radio.toggled.connect(self.phase_radio_toggled)

        #layout - groupBox for fft info
        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(self.ampl_radio)
        group_box_layout.addWidget(self.phase_radio)
        group_box_layout.addWidget(fft_coef_label)
        group_box_layout.addWidget(fft_coef_list)

        params_group_box = QGroupBox()
        params_group_box.setLayout(group_box_layout)

        layout = QVBoxLayout()
        child_layout = QHBoxLayout()

        layout.addLayout(child_layout)

        child_layout.addWidget(self.chart_view)
        child_layout.addWidget(params_group_box)

        self.setLayout(layout)

        self.ampl_radio.setChecked(True)

    def ampl_radio_toggled(self, state):
        if state:
            self.phase_radio.setChecked(False)
            self.chart_view.build_plot(self.ampl_data, "Частотно-временой анализ", "Амплитуда", "Частота, Гц")

    def phase_radio_toggled(self, state):
        if state:
            self.ampl_radio.setChecked(False)
            self.chart_view.build_plot(self.phase_data, "Частотно-временой анализ", "Фаза", "Частота, Гц")


