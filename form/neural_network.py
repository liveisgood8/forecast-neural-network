from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QGroupBox, QLabel

from modules.DataProcessing import DataProcessing

class NeuralNetworkDialog(QDialog):

    def __init__(self, data_processing : DataProcessing, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Окно настройки нейронной сети')
        self.setMinimumSize(600, 900)

        self.data_processing = data_processing

        self.data_processing.normalize_data()


