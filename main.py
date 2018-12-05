import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from form import form_design
from form.graphic_form import GraphicWindow
from data.imces import build_url, load_data, get_detectors_sn
from data.dictionaries import *
from data.parser import DataParser

'''
================================================================
TODO:

1. Перевести parser на pandas
================================================================
'''

class MApplication(QtWidgets.QMainWindow, form_design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #Перенесем окно в центр
        qt_rect = self.frameGeometry()
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()
        qt_rect.moveCenter(center_point)
        self.move(qt_rect.topLeft())

        #Изменение данных в комбобоксах
        self.paramCombo.currentTextChanged.connect(self.fill_stations)
        self.stationCombo.currentTextChanged.connect(self.fill_detectors)

        #Клик по кнопке получения данных
        self.getDataButton.clicked.connect(self.get_data_btn_clicked)

        #Клик по импорту данных
        self.import_action.triggered.connect(self.import_data)

        self.fill_params()

    #Получение данных
    #Формат даты и времени: 2018-04-10T16:31:00+07
    def get_data_btn_clicked(self):
        q_start_datetime, start_datetime_str = self.make_datetime(self.startDateControl.dateTime())
        q_end_datetime, end_datetime_str = self.make_datetime(self.endDateControl.dateTime())

        param_name = rus_to_id_dict[self.paramCombo.currentText()]
        param_id = id_dict[param_name]
        station_id = int(self.stationCombo.currentText())
        detector_sn = self.detectorCombo.currentText()

        urlfinal = build_url(start_datetime_str, end_datetime_str, param_id, station_id, detector_sn)

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        data = load_data(urlfinal)
        QGuiApplication.setOverrideCursor(Qt.ArrowCursor)

        dataparser = DataParser(self.paramCombo.currentText(), param_name)
        dataparser.set_station(station_id)
        dataparser.set_detector(detector_sn)
        dataparser.set_data(data)

        #Если таких данные нет, покажем предупреждение
        if dataparser.is_data_empty():
            self.show_msgbox('Данных за указанный интервал времени не найдено!', True)
        else:
            #Проверим диапазон дат и если он различный, то выведем предупреждение
            (is_date_diff, orig_start_date_str, orig_end_date_str) = dataparser.compare_date(q_start_datetime, q_end_datetime)
            if is_date_diff:
                self.show_msgbox('Доступный временной интервал отличается от заданого!\n'
                                 + 'Данные будут отображены за следующий интервал:\n'
                                 + orig_start_date_str + ' - ' + orig_end_date_str)

            graphic_wnd = GraphicWindow(dataparser, self)
            graphic_wnd.exec()

    #Функции для заполнения чекбоксов
    def fill_params(self):
        for param in rus_to_id_dict.keys():
            self.paramCombo.addItem(param)

    def fill_stations(self, paramComboText):
        param_name = rus_to_id_dict[paramComboText]

        self.stationCombo.clear()
        for station_id in station_dict[param_name]:
            self.stationCombo.addItem(str(station_id))

    def fill_detectors(self, stationComboText):
        param_name = rus_to_id_dict[self.paramCombo.currentText()]

        self.detectorCombo.clear()

        if stationComboText != '':
            for detectorName in get_detectors_sn(param_name, int(stationComboText)):
                self.detectorCombo.addItem(detectorName)

    #Convert DateTime by timezone and to string
    def make_datetime(self, datetime):
        q_datetime = datetime.addSecs(7*60*60)
        datetime_str = q_datetime.toString("yyyy-MM-ddThh:mm:ss+07")

        return q_datetime, datetime_str

    def show_msgbox(self, text, error=False):
        msg = QtWidgets.QMessageBox()

        if not error:
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle('Внимание')
        else:
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setWindowTitle('Ошибка')

        msg.setText(text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec()

    def import_data(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')

        if not fname[0]:
            return

        data_parser = DataParser()
        data_parser.import_from_file(fname[0])

        graphic_wnd = GraphicWindow(data_parser, self)
        graphic_wnd.exec()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MApplication()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()


