from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QGroupBox, QLabel
from PyQt5.QtChart import QLineSeries, QValueAxis, QChart
from PyQt5.QtCore import Qt

from modules.ChartView import ChartView


class FftDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Преобразование фурье')
        self.setMinimumSize(1200, 800)

        #Виджет с графиком
        self.chart_view = ChartView()
        self.chart_view.setMinimumSize(1000, 800)

        #Виджет для коэффициентов
        self.fft_coef_list_wdgt = QListWidget()
        fft_coef_label = QLabel('Коэффициенты фурье:')

        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(fft_coef_label)
        group_box_layout.addWidget(self.fft_coef_list_wdgt)

        params_group_box = QGroupBox()
        params_group_box.setLayout(group_box_layout)

        layout = QVBoxLayout()
        child_layout = QHBoxLayout()

        layout.addLayout(child_layout)

        child_layout.addWidget(self.chart_view)
        child_layout.addWidget(params_group_box)

        self.setLayout(layout)

    def build_plot(self, data):
        lineSeries = QLineSeries()
        lineSeries.setPointsVisible(True)

        xf = data[0]
        yf = data[1]

        for i in range(len(yf)):
            lineSeries.append(xf[i], yf[i])

            # Заполним лист с коэффициентами Фурье
            self.fft_coef_list_wdgt.addItem(str(yf[i]))

        chart = QChart()
        chart.addSeries(lineSeries)
        chart.setTitle("Частотно-временой анализ")
        chart.legend().hide()

        axisX = QValueAxis()
        axisX.setTitleText("Частота, Гц")
        axisX.setLabelsAngle(-60)
        axisX.setTickCount(20)

        axisY = QValueAxis()
        axisY.setTitleText("Амплитуда")
        axisY.setTickCount(20)

        chart.addAxis(axisX, Qt.AlignBottom)
        chart.addAxis(axisY, Qt.AlignLeft)
        lineSeries.attachAxis(axisX)
        lineSeries.attachAxis(axisY)

        self.chart_view.setChart(chart)



