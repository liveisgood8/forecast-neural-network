from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import *
from keras.utils import plot_model
from keras import backend as KerasBackend
from math import sqrt
from abc import ABCMeta, abstractmethod
import numpy

from PyQt5 import QtCore

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

#Make scaler [-1; 1]
def make_scaler():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler

def scale(data):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data)
    # transform train
    data = data.reshape(data.shape[0], data.shape[1])
    data_scaled = scaler.transform(data)
    return scaler, data_scaled

def scale_all(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0, 0]


class Predictions:
    values = None
    rmse = 0
    train_time = 0


class TimeHistory(Callback):
    start_time = 0
    end_time = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()

    def get_time_delta(self):
        return self.end_time - self.start_time


class GuiController(Callback):
    terminate = False
    callback_func = None
    epoch_num = 0

    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if self.terminate:
            self.model.stop_training = True

        if self.callback_func:
            self.callback_func(self.epoch_num)
            self.epoch_num += 1

    def on_train_end(self, logs=None):
        self.epoch_num = 0


class INetwork:
    __metaclass__ = ABCMeta
    RMSE_ABORTED_VALUE = list()

    def __init__(self, data, repeats, epoch, batch_size, lstm_neurons, lstm_layers, optimizer):
        self.repeats = repeats
        self.epoch = epoch
        self.batch_size = batch_size
        self.lstm_neurons = lstm_neurons
        self.raw_values = data.ix[:, 1].tolist()
        self.gui_controller = GuiController()

        self.lstm_layers = lstm_layers
        self.optimizer = optimizer
        self.train = None
        self.train_scaled = None
        self.scaler = None

        KerasBackend.clear_session()

    def make_model(self, batch_size, inp_shape_dim, inp_shape_ddim):
        model = Sequential()

        for i in range(self.lstm_layers - 1):
            model.add(LSTM(self.lstm_neurons, batch_input_shape=(batch_size, inp_shape_dim, inp_shape_ddim),
                           stateful=True, return_sequences=True))
            model.add(Dropout(0.2))

        model.add(LSTM(self.lstm_neurons, batch_input_shape=(batch_size, inp_shape_dim, inp_shape_ddim), stateful=True))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        plot_model(model, to_file='model_single.png', show_layer_names=True, show_shapes=True)

        return model

    def make_multi_predictions(self, repeat_iterator_callback, epoch_iterator_callback):
        for i in range(self.repeats):
            model, train_time = self.fit_lstm(epoch_iterator_callback)

            predictions = Predictions()
            predictions.train_time = train_time
            predictions.values, predictions.rmse = self.prediciotns_repeat(model)
            repeat_iterator_callback(i, predictions)

    @abstractmethod
    def fit_lstm(self, iteration_callback): raise NotImplementedError

    @abstractmethod
    def prediciotns_repeat(self, lstm_model): raise NotImplementedError

    @abstractmethod
    def prepare_data(self): raise NotImplementedError


class NSingleStep(INetwork):
    def __init__(self, data, train_size, repeats, epoch, batch_size, lstm_neurons, lstm_layers, optimizer):
        super().__init__(data, repeats, epoch, batch_size, lstm_neurons, lstm_layers, optimizer)

        self.train_size = train_size
        self.test = None
        self.test_scaled = None

        self.prepare_data()

    def prepare_data(self):
        # transform data to be supervised learning
        supervised = timeseries_to_supervised(self.raw_values, 1)
        supervised_values = supervised.values

        # split data into train and test-sets
        train, test = supervised_values[:self.train_size], supervised_values[self.train_size:]

        # transform the scale of the data
        self.scaler, self.train_scaled, self.test_scaled = scale_all(train, test)

    # fit an LSTM network to training data
    def fit_lstm(self, iteration_callback):
        X, y = self.train_scaled[:, 0:-1], self.train_scaled[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        Xt, yt = self.test_scaled[:, 0:-1], self.test_scaled[:, -1]
        Xt = Xt.reshape(Xt.shape[0], 1, Xt.shape[1])

        # Callbacks
        # early_stop = EarlyStoppingByLossVal(monitor='val_loss', value=0.05, verbose=1)
        # early_stop = EarlyStopping(monitor='val_loss', patience=10,  verbose=1)
        # check_point = ModelCheckpoint(filepath='model-{epoch:03d}.h5', monitor='val_loss', verbose=1, mode='min',
        #                               save_best_only=True, period=50)

        train_timer = TimeHistory()
        self.gui_controller.callback_func = iteration_callback

        model = self.make_model(self.batch_size, X.shape[1], X.shape[2])
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        model.fit(X, y, epochs=self.epoch, batch_size=self.batch_size, verbose=0, shuffle=False,
                  callbacks=[train_timer, self.gui_controller],
                  validation_data=(Xt, yt))

        #Для разных batch_size при обучении и предсказании
        predict_model = self.make_model(1, X.shape[1], X.shape[2])

        old_weights = model.get_weights()
        predict_model.set_weights(old_weights)
        predict_model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        return predict_model, train_timer.get_time_delta()

    def prediciotns_repeat(self, lstm_model):
        # walk-forward validation on the test data
        predictions = list()
        for i in range(len(self.test_scaled)):
            if self.gui_controller.terminate:
                break

            # make one-step forecast
            X = self.test_scaled[i, 0:-1]
            yhat = forecast_lstm(lstm_model, 1, X)

            # invert scaling
            yhat = invert_scale(self.scaler, X, yhat)

            # store forecast
            predictions.append(yhat)

        rmse_value = sqrt(mean_squared_error(self.raw_values[self.train_size:], predictions))

        ## Lets make one-step prediction in future
        X = self.test_scaled[-1:, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        yhat = invert_scale(self.scaler, X, yhat)
        predictions.append(yhat)

        if not self.gui_controller.terminate:
            # report performance
            rmse = list()
            rmse.append(rmse_value)
            return predictions, rmse
        else:
            return predictions, self.RMSE_ABORTED_VALUE


class NMultiWindowMode(INetwork):
    #TODO Доделать оконный метод

    def __init__(self, data, start_position, repeats, epoch, batch_size,
                 lstm_neurons, lstm_layers, optimizer, window_size):
        super().__init__(data, repeats, epoch, batch_size, lstm_neurons, lstm_layers, optimizer)

        self.test_scaled = list()
        self.window_size = window_size
        self.start_position = start_position
        self.train_data_x = None
        self.train_data_y = None
        self.test_data_x = None
        self.test_data_y = None

        self.prepare_data()

    def prepare_data(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        np_array = numpy.array(self.raw_values).reshape(len(self.raw_values), 1)

        data_scaled = self.scaler.fit_transform(np_array)
        data_scaled = data_scaled.reshape(len(data_scaled))
        # data_scaled = self.raw_values

        train_data_raw = data_scaled[0:self.start_position]
        test_data_raw = data_scaled[self.start_position:]

        self.train_data_x, self.train_data_y = self.create_dataset(train_data_raw)

        self.test_data_x, self.test_data_y = self.create_dataset(test_data_raw)

    def prediciotns_repeat(self, lstm_model):
        # walk-forward validation on the test data
        predictions_scaled = lstm_model.predict(self.test_data_x)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        predictions = predictions.reshape(len(predictions))
        # predictions = predictions_scaled

        if not self.gui_controller.terminate:
            # report performance
            rmse = list()
            rmse.append(sqrt(mean_squared_error(self.raw_values[self.start_position + self.window_size + 1:],
                                                predictions)))
            return predictions, rmse
        else:
            return predictions, self.RMSE_ABORTED_VALUE

    def fit_lstm(self, iteration_callback):
        train_timer = TimeHistory()
        self.gui_controller.callback_func = iteration_callback

        model = Sequential()
        model.add(Dense(12, input_dim=self.window_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        model.fit(self.train_data_x, self.train_data_y, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                  callbacks=[train_timer, self.gui_controller])

        return model, train_timer.get_time_delta()

    # convert an array of values into a dataset matrix
    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.window_size - 1):
            a = dataset[i:(i + self.window_size)]
            dataX.append(a)
            dataY.append(dataset[i + self.window_size])
        return numpy.array(dataX), numpy.array(dataY)


class NeuralNetworkTeacher(QtCore.QThread):
    signal_epoch = QtCore.pyqtSignal(int)
    signal_repeat = QtCore.pyqtSignal(int, Predictions)
    signal_complete = QtCore.pyqtSignal()

    def __init__(self, neural_network: INetwork, parent=None):
        super().__init__(parent)
        self.neural_network = neural_network

    def tterminate(self):
        self.neural_network.gui_controller.terminate = True

    def run(self):
        self.neural_network.make_multi_predictions(lambda i, predictions: self.signal_repeat.emit(i, predictions),
                                                   lambda i: self.signal_epoch.emit(i))
        self.signal_complete.emit()
