from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QGroupBox, QLabel

from modules.DataAnalyzer import DataAnalyzer

class NeuralNetworkDialog(QDialog):

    def __init__(self, data_analyzer : DataAnalyzer, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Окно настройки нейронной сети')
        self.setMinimumSize(600, 900)

        self.data_analyzer = data_analyzer


