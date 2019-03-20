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

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference_by_prev(yhat, prev):
    return yhat + prev

#Make scaler [-1; 1]
def make_scaler():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler

def scale(scaler, data):
    scaler = scaler.fit(data)

    # transform
    data = data.reshape(data.shape[0], data.shape[1])
    data_scaled = scaler.transform(data)
    return data_scaled

def scale_test(train, test):
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


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class INetwork:
    __metaclass__ = ABCMeta
    RMSE_ABORTED_VALUE = -1

    def __init__(self, data, repeats, epoch, batch_size, lstm_neurons, lstm_layers, optimizer):
        self.repeats = repeats
        self.epoch = epoch
        self.batch_size = batch_size
        self.lstm_neurons = lstm_neurons
        self.raw_values = data.ix[:, 1].tolist()

        self.lstm_layers = lstm_layers
        self.optimizer = optimizer
        self.train = None
        self.train_scaled = None

        self.scaler = make_scaler()
        self.gui_controller = GuiController()

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

        return model

    # fit an LSTM network to training data
    def fit_lstm(self, iteration_callback):
        X, y = self.train_scaled[:, 0:-1], self.train_scaled[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        check_point = ModelCheckpoint(filepath='model-{epoch:03d}.h5', monitor='val_loss', verbose=1, mode='min',
                                      save_best_only=True, period=50)

        train_timer = TimeHistory()

        model = self.make_model(self.batch_size, X.shape[1], X.shape[2])
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        model.fit(X, y, epochs=self.epoch, batch_size=self.batch_size, verbose=1, shuffle=False,
                  callbacks=[check_point, train_timer])

        #Для разных batch_size при обучении и предсказании
        predict_model = self.make_model(1, X.shape[1], X.shape[2])

        old_weights = model.get_weights()
        predict_model.set_weights(old_weights)
        predict_model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        # model = load_model('model-200.h5')

        return predict_model, train_timer.get_time_delta()

    def make_multi_predictions(self, repeat_iterator_callback, epoch_iterator_callback):
        for i in range(self.repeats):
            model, train_time = self.fit_lstm(epoch_iterator_callback)

            predictions = Predictions()
            predictions.train_time = train_time
            predictions.values, predictions.rmse = self.prediciotns_repeat(model)
            repeat_iterator_callback(i, predictions)

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
        self.scaler, self.train_scaled, self.test_scaled = scale_test(train, test)

    # fit an LSTM network to training data
    def fit_lstm(self, iteration_callback):
        X, y = self.train_scaled[:, 0:-1], self.train_scaled[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        Xt, yt = self.test_scaled[:, 0:-1], self.test_scaled[:, -1]
        Xt = Xt.reshape(Xt.shape[0], 1, Xt.shape[1])

        # Callbacks
        # early_stop = EarlyStoppingByLossVal(monitor='val_loss', value=0.05, verbose=1)
        # early_stop = EarlyStopping(monitor='val_loss', patience=150, min_delta=0.1, verbose=1)
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

        if not self.gui_controller.terminate:
            # report performance
            rmse = sqrt(mean_squared_error(self.raw_values[self.train_size:], predictions))
            return predictions, rmse
        else:
            return predictions, self.RMSE_ABORTED_VALUE


class NMultiWindowMode(INetwork):
    #TODO Доделать оконный метод
    RMSE_SKIP = -2

    def __init__(self, data, predictions_num, repeats, epoch, batch_size, lstm_neurons, window_size):
        super().__init__(data, repeats, epoch, batch_size, lstm_neurons)

        self.predictions_num = predictions_num
        self.test_scaled = list()
        self.window_size = window_size
        # self.raw_values_window = self.raw_values[len(self.raw_values) - self.window_size:]
        self.raw_values_window = self.raw_values

        self.prepare_data()

    def prepare_data(self):
        # transform data to be stationary
        diff_values = difference(self.raw_values_window, 1)

        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        self.train = supervised_values[0:1]
        self.train_scaled = scale(self.scaler, self.train)

    def prediciotns_repeat(self, lstm_model):
        # walk-forward validation on the test data
        predictions = list()

        for i in range(self.predictions_num):
            if self.gui_controller.terminate:
                break

            prev_row_value = self.raw_values_window[-1]
            prev_value = self.train_scaled[-1, 0:-1]

            yhat = forecast_lstm(lstm_model, 1, prev_value)
            yhat = invert_scale(self.scaler, prev_value, yhat)
            yhat_converted = self.inverse_difference_byprev(prev_row_value, yhat)
            predictions.append(yhat_converted)

            self.raw_values_window = self.raw_values_window[1:]
            self.raw_values_window.append(yhat_converted)
            self.prepare_data()
            lstm_model = self.fit_lstm(None)

        return predictions, self.RMSE_SKIP


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
