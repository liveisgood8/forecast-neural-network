import os

from PyQt5.QtChart import QChartView, QChart, QLineSeries, QDateTimeAxis, QValueAxis
from PyQt5.Qt import *


class ChartView(QChartView):

    def __init__(self, no_margins=False):
        super().__init__()
        self.__last_mouse_pos = None
        self.lineSeries = QLineSeries()
        self.no_margins = no_margins

        #Create a menu by right click on QChartView
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        ##Axis cofnig
        self.x_time_scaled = False
        self.x_name = None
        self.y_name = None
        self.x_tick_num = 30
        self.y_tick_num = 20
        self.x_label_angle = -90
        self.y_label_angle = 0
        self.y_min = None
        self.y_max = None
        self.x_min = None
        self.x_max = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self.__last_mouse_pos = event.pos()
            event.accept()

        QChartView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MiddleButton:
            dif_pos = event.pos() - self.__last_mouse_pos

            self.chart().scroll(-dif_pos.x(), dif_pos.y())
            event.accept()

            self.__last_mouse_pos = event.pos()

        QChartView.mouseMoveEvent(self, event)

    def wheelEvent(self, event: QWheelEvent):
        factor = event.angleDelta().y()
        if factor < 0:
            self.chart().zoom(0.75)
        else:
            self.chart().zoom(1.25)

    def show_context_menu(self, pos : QPoint):
        menu = QMenu()

        plot_save_action = QAction('Сохранить график', self)
        plot_save_action.triggered.connect(self.save_plot)
        menu.addAction(plot_save_action)

        menu.exec(self.mapToGlobal(pos))

    def save_plot(self):
        pixmap = self.grab()

        pixmap_name = QFileDialog.getSaveFileName(self, 'Сохранение графика', None, "PNG (*.png);;JPG (*.jpg);;BMP (*.bmp)")[0]

        if not pixmap_name:
            return

        pixmap.save(pixmap_name, os.path.splitext(pixmap_name)[1][1:])

    def make_axis(self):
        if self.x_time_scaled:
            axis_x = QDateTimeAxis()
            axis_x.setFormat("yyyy-MM-dd HH:mm:ss")
            axis_x.setTitleText("Время")
        else:
            axis_x = QValueAxis()
            axis_x.setTitleText(self.x_name)

        if self.x_min:
            axis_x.setMin(self.x_min)
        if self.x_max:
            axis_x.setMax(self.x_max)

        axis_x.setLabelsAngle(self.x_label_angle)
        axis_x.setTickCount(self.x_tick_num)

        axis_y = QValueAxis()
        if self.y_min:
            axis_y.setMin(self.y_min)
        if self.y_max:
            axis_y.setMax(self.y_max)

        axis_y.setTitleText(self.y_name)
        axis_y.setTickCount(self.y_tick_num)
        axis_y.setLabelsAngle(self.y_label_angle)

        return axis_x, axis_y

    def build_plot(self, data, title):
        axis_x, axis_y = self.make_axis()

        chart = QChart()
        if self.no_margins:
            chart.setMargins(QMargins(0, 0, 0, 0))

        self.clean()
        self.lineSeries.setPointsVisible(True)

        if data != None:
            xf = data[0]
            yf = data[1]

            for i in range(len(xf)):
                self.lineSeries.append(xf[i], yf[i])

        chart.addSeries(self.lineSeries)
        chart.legend().hide()
        chart.setTitle(title)
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)

        self.lineSeries.attachAxis(axis_x)
        self.lineSeries.attachAxis(axis_y)

        self.setChart(chart)

    def series_append(self, x, y, x_min, y_min, x_range=False, y_range=False):
        self.lineSeries.append(x, y)

        self.chart().axisX().setMin(x_min)
        self.chart().axisY().setMin(y_min - 0.2)

        if x_range:
            self.chart().axisX().setMax(x)

        if y_range:
            self.chart().axisY().setMax(y + 0.2)

    def clean(self):
        self.lineSeries.clear()

