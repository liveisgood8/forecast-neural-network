from PyQt5 import QtWidgets

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