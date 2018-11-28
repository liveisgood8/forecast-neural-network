from PyQt5.QtChart import *
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.Qt import *

class ChartView(QChartView):

    def __init__(self):
        super().__init__()
        self.__last_mouse_pos = None

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