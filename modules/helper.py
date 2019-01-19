from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel


def show_msgbox(text, error=False):
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


def grid_add_label_widget(layout, text, widget, row):
    layout.addWidget(QLabel(text + ':'), row, 0)
    layout.addWidget(widget, row, 1)

    return row + 1
