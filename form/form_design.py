# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form_design.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 250)
        MainWindow.setMinimumSize(QtCore.QSize(500, 250))
        MainWindow.setMaximumSize(QtCore.QSize(500, 250))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 10, 481, 161))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.paramLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.paramLabel.setObjectName("paramLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.paramLabel)
        self.paramCombo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.paramCombo.setObjectName("paramCombo")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.paramCombo)
        self.stationLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.stationLabel.setObjectName("stationLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.stationLabel)
        self.stationCombo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.stationCombo.setObjectName("stationCombo")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.stationCombo)
        self.detectorLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.detectorLabel.setObjectName("detectorLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.detectorLabel)
        self.detectorCombo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.detectorCombo.setObjectName("detectorCombo")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.detectorCombo)
        self.startDateLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.startDateLabel.setObjectName("startDateLabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.startDateLabel)
        self.startDateControl = QtWidgets.QDateTimeEdit(self.formLayoutWidget)
        self.startDateControl.setDate(QtCore.QDate(2018, 1, 1))
        self.startDateControl.setObjectName("startDateControl")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.startDateControl)
        self.endDateLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.endDateLabel.setObjectName("endDateLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.endDateLabel)
        self.endDateControl = QtWidgets.QDateTimeEdit(self.formLayoutWidget)
        self.endDateControl.setDate(QtCore.QDate(2018, 1, 1))
        self.endDateControl.setObjectName("endDateControl")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.endDateControl)
        self.getDataButton = QtWidgets.QPushButton(self.centralwidget)
        self.getDataButton.setGeometry(QtCore.QRect(320, 180, 171, 25))
        self.getDataButton.setObjectName("getDataButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.import_action = QtWidgets.QAction(MainWindow)
        self.import_action.setObjectName("import_action")
        self.menu.addAction(self.import_action)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Анализ данных"))
        self.paramLabel.setText(_translate("MainWindow", "Параметр измерения:"))
        self.stationLabel.setText(_translate("MainWindow", "Выбор станции"))
        self.detectorLabel.setText(_translate("MainWindow", "Выбор датчика"))
        self.startDateLabel.setText(_translate("MainWindow", "Дата начала:"))
        self.endDateLabel.setText(_translate("MainWindow", "Дата окончания:"))
        self.getDataButton.setText(_translate("MainWindow", "Получить данные"))
        self.menu.setTitle(_translate("MainWindow", "Файл"))
        self.import_action.setText(_translate("MainWindow", "Импорт данных"))

