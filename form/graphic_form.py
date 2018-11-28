from PyQt5.QtWidgets import QDialog, QGridLayout, QVBoxLayout, QLabel, QComboBox, QGroupBox
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenuBar, QAction, QMessageBox, QPushButton, QFileDialog
from PyQt5.QtChart import QLineSeries, QDateTimeAxis, QValueAxis, QChart

from data.parser import DataParser
from modules.ChartView import ChartView
from modules.DataAnalyzer import DataAnalyzer
from form.fft_form import FftDialog


class GraphicWindow(QDialog):

    def __init__(self, parser : DataParser, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Окно графиков')
        self.setMinimumSize(1200, 800)

        self.parser = parser
        self.data_analyzer = DataAnalyzer()

        self.chart_view = ChartView()
        self.chart_view.setMinimumSize(1000, 800)

        #Словарь для хранения лейблов со значениеями статистических характеристик
        self.param_label_container = {}

        #Список параметров для построения
        self.combo_headers = QComboBox()

        #Кнопка для построения Преобр. Фурье
        fft_plot_button = QPushButton()
        fft_plot_button.setText('Преобразование Фурье')
        fft_plot_button.clicked.connect(self.fft_build_plot)

        #group box layout
        plot_controls_layout = QVBoxLayout()
        plot_controls_layout.addWidget(self.combo_headers, 0, Qt.AlignTop)
        plot_controls_layout.addWidget(fft_plot_button, 0, Qt.AlignTop)

        group_box_plot = QGroupBox()
        group_box_plot.setLayout(plot_controls_layout)
        group_box_plot.setTitle('Параметры построения графика')

        #param group box
        params_layout = QGridLayout()

        self.add_statistics_parm('Минимальное значение', params_layout)
        self.add_statistics_parm('Максимальное значение', params_layout)
        self.add_statistics_parm('Мат. ожидание', params_layout)
        self.add_statistics_parm('Дисперсия', params_layout)
        self.add_statistics_parm('Медиана', params_layout)
        self.add_statistics_parm('Мода', params_layout)
        self.add_statistics_parm('Размах', params_layout)

        #Статистические параметры
        group_box_param = QGroupBox()
        group_box_param.setTitle('Статистические характеристики')
        group_box_param.setLayout(params_layout)

        #Общий слой для всех настроек
        all_param_layout = QVBoxLayout()
        all_param_layout.addWidget(group_box_plot)
        all_param_layout.addWidget(group_box_param)

        #Дочерний слой
        layout = QGridLayout()
        layout.addWidget(self.chart_view, 0, 0)
        layout.addLayout(all_param_layout, 0, 1, Qt.AlignTop)

        #Основной слой
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)

        self.setLayout(main_layout)

        #Сигнал о изменении текста comboBox с заголовками данных
        self.combo_headers.currentTextChanged.connect(self.buildplot)

        #Заполним comboBox
        for header in parser.get_headers():
            self.combo_headers.addItem(header)

        #Создадим меню
        menu_bar = QMenuBar()

        file_menu = menu_bar.addMenu('Файл')

        plot_save_action = QAction('Сохранить график', self)
        plot_save_action.triggered.connect(self.save_plot)
        file_menu.addAction(plot_save_action)

        export_data_action = QAction('Экспортировать данные', self)
        export_data_action.triggered.connect(self.export_data)
        file_menu.addAction(export_data_action)

        main_layout.setMenuBar(menu_bar)

    def add_statistics_parm(self, name, layout: QGridLayout):
        param_label = QLabel()
        param_label.setText(name + ':')

        param_value_label = QLabel()

        #Номер последнего параметра в списке
        self.param_label_container[name] = param_value_label
        numofparam = len(self.param_label_container)

        layout.addWidget(param_label, numofparam - 1, 0)
        layout.addWidget(param_value_label, numofparam - 1, 1)

    def buildplot(self, header_name):
        #Строим график
        lineSeries = QLineSeries()
        lineSeries.setPointsVisible(True)

        #Получим необходимую колонку
        data = self.parser.get_column(header_name)

        #Занесем данные в анализатор
        self.data_analyzer.set_data(data)

        #Конвертируем в удобный вид
        converted_data = self.data_analyzer.convert_data()

        for key in converted_data:
            lineSeries.append(key, converted_data[key])

        chart = QChart()
        chart.addSeries(lineSeries)
        chart.setTitle(self.parser.rus_param_name)
        chart.legend().hide()

        axisX = QDateTimeAxis()
        axisX.setTickCount(20)
        axisX.setFormat("yyyy-MM-dd HH:mm:ss");
        axisX.setTitleText("Время")
        axisX.setLabelsAngle(-90)

        axisY = QValueAxis()
        axisY.setTitleText(self.combo_headers.currentText())
        axisY.setMin(self.data_analyzer.min() - 0.1)
        axisY.setMax(self.data_analyzer.max() + 0.1)
        axisY.setTickCount(15)

        chart.addAxis(axisX, Qt.AlignBottom)
        chart.addAxis(axisY, Qt.AlignLeft)
        lineSeries.attachAxis(axisX)
        lineSeries.attachAxis(axisY)

        self.chart_view.setChart(chart)

        #Заполним параметры
        self.param_label_container['Минимальное значение'].setText(str(self.data_analyzer.min()))
        self.param_label_container['Максимальное значение'].setText(str(self.data_analyzer.max()))
        self.param_label_container['Мат. ожидание'].setText(str(self.data_analyzer.mean()))
        self.param_label_container['Дисперсия'].setText(str(self.data_analyzer.std()))
        self.param_label_container['Медиана'].setText(str(self.data_analyzer.median()))
        self.param_label_container['Мода'].setText(str(self.data_analyzer.mode()))
        self.param_label_container['Размах'].setText(str(self.data_analyzer.min_max_delta()))

        #TMP
        #print(self.data_analyzer.fft())

    def fft_build_plot(self):
        #self.data_analyzer.fft()
        fft_dlg = FftDialog()
        fft_dlg.build_plot(self.data_analyzer.fft())
        fft_dlg.exec()

    def save_plot(self):
        pixmap = self.chart_view.grab()

        pixmap.save("Plot", "PNG")

        message = QMessageBox()
        message.setWindowTitle('Внимание')
        message.setIcon(QMessageBox.Information)
        message.setText('График успешно сохранен!')
        message.exec()

    def export_data(self):
        file_name = QFileDialog.getSaveFileName(self, 'Save file', None, "All Files (*)")

        if file_name[0]:
            self.parser.export_to_file(file_name[0])

