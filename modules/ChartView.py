from PyQt5.QtChart import QChartView, QChart, QLineSeries, QDateTimeAxis, QValueAxis
from PyQt5.Qt import *


class ChartView(QChartView):

    def __init__(self):
        super().__init__()
        self.__last_mouse_pos = None

        #Create a menu by right click on QChartView
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)


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

        pixmap.save("Plot.png", "PNG")

        message = QMessageBox()
        message.setWindowTitle('Внимание')
        message.setIcon(QMessageBox.Information)
        message.setText('График успешно сохранен!')
        message.exec()

    def build_plot(self, data, title, y_name, x_name=None, set_x_time_scaled=False, y_min=None, y_max=None):
        chart = QChart()

        lineSeries = QLineSeries()
        lineSeries.setPointsVisible(True)

        xf = data[0]
        yf = data[1]

        for i in range(len(xf)):
            lineSeries.append(xf[i], yf[i])

        chart.addSeries(lineSeries)
        chart.legend().hide()
        chart.setTitle(title)

        if set_x_time_scaled:
            axis_x = QDateTimeAxis()
            axis_x.setFormat("yyyy-MM-dd HH:mm:ss");
            axis_x.setTitleText("Время")
            axis_x.setLabelsAngle(-90)
        else:
            axis_x = QValueAxis()
            axis_x.setTitleText(x_name)
            axis_x.setLabelsAngle(-60)

        axis_x.setTickCount(20)

        axis_y = QValueAxis()
        axis_y.setTitleText(y_name)
        axis_y.setTickCount(20)
        if y_min:
            axis_y.setMin(y_min)
        if y_max:
            axis_y.setMax(y_max)

        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        lineSeries.attachAxis(axis_x)
        lineSeries.attachAxis(axis_y)

        self.setChart(chart)
