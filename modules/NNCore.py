from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as KerasBackend
from math import sqrt
from matplotlib import pyplot
import numpy

from PyQt5 import QtCore

class Predictions:
    values = None
    rmse = 0


class NeuralNetwork:
    m_terminate = False #Response for interupting prediction
    RMSE_ABORTED_VALUE = -1

    def __init__(self, data, train_size, predictions_num, repeats, epoch, batch_size, lstm_neurons):
        self.repeats = repeats
        self.epoch = epoch
        self.batch_size = batch_size
        self.lstm_neurons = lstm_neurons
        self.train_size = train_size
        self.predictions_num = predictions_num
        self.raw_values = data.ix[:, 1]

        if len(self.raw_values) == self.train_size:
            self.is_future = True #Прогноз на будущее, за пределами допустимых значений
        else:
            self.is_future = False

        # transform data to be stationary
        diff_values = self.difference(self.raw_values, 1)

        # transform data to be supervised learning
        supervised = self.timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        self.scaler = self.make_scaler()

        if self.is_future:
            self.train = supervised_values[0:train_size]
            self.train_scaled = self.scale(self.scaler, self.train)
        else:
            # split data into train and test-sets on two equal parts
            self.train, self.test = supervised_values[0:train_size], \
                                    supervised_values[-train_size:-train_size + predictions_num]

            # transform the scale of the data
            self.train_scaled, self.test_scaled = self.scale(self.scaler, self.train), \
                                                  self.scale(self.scaler, self.test)

        KerasBackend.clear_session()

    # frame a sequence as a supervised learning problem
    @staticmethod
    def timeseries_to_supervised(data, lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df

    # create a differenced series
    @staticmethod
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    # invert differenced value
    @staticmethod
    def inverse_difference(history, yhat, interval=1):
        return yhat + history.iloc[-interval]

    @staticmethod
    def inverse_difference_byprev(yprev, yhat):
        return yhat + yprev

    #Make scaler [-1; 1]
    def make_scaler(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler

    # scale data by using scaler
    @staticmethod
    def scale(scaler, data):
        scaler = scaler.fit(data)
        # transform
        data = data.reshape(data.shape[0], data.shape[1])
        data_scaled = scaler.transform(data)
        return data_scaled

    # inverse scaling for a forecasted value
    @staticmethod
    def invert_scale(scaler, X, value):
        new_row = [x for x in X] + [value]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    # fit an LSTM network to training data
    def fit_lstm(self, iteration_callback):
        X, y = self.train_scaled[:, 0:-1], self.train_scaled[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(self.lstm_neurons, batch_input_shape=(self.batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        for i in range(self.epoch):
            if self.m_terminate:
                break

            model.fit(X, y, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=False)
            model.reset_states()
            iteration_callback(i)
        return model

    # make a one-step forecast
    def forecast_lstm(self, model, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        return yhat[0, 0]

    def make_multi_predictions(self, repeat_iterator_callback, epoch_iterator_callback):
        for i in range(self.repeats):
            if self.m_terminate:
                break

            model = self.fit_lstm(epoch_iterator_callback)

            predictions = Predictions()
            if self.is_future:
                predictions.values, predictions.rmse = self.future_predict(model)
            else:
                predictions.values, predictions.rmse = self.prediciotns_repeat(model)

            repeat_iterator_callback(i, predictions)

    def prediciotns_repeat(self, lstm_model):

        # # forecast the entire training dataset to build up state for forecasting - ???????
        # train_reshaped = self.train_scaled[:, 0].reshape(len(self.train_scaled), 1, 1)
        # lstm_model.predict(train_reshaped, batch_size=1)

        # walk-forward validation on the test data
        predictions = list()
        for i in range(len(self.test_scaled)):
            if self.m_terminate:
                break

            # make one-step forecast
            X, y = self.test_scaled[i, 0:-1], self.test_scaled[i, -1]
            yhat = self.forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = self.invert_scale(self.scaler, X, yhat)
            # invert differencing
            yhat = self.inverse_difference(self.raw_values, yhat, len(self.test_scaled) + 1 - i)
            # store forecast
            predictions.append(yhat)
            expected = self.raw_values[len(self.train) + i + 1]
            # print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

        if not self.m_terminate:
            # report performance
            rmse = sqrt(mean_squared_error(self.raw_values[-self.train_size:-self.train_size + self.predictions_num]
                                           .values.tolist(), predictions))
            # print('Test RMSE: %.3f' % rmse)

            # line plot of observed vs predicted
            # pyplot.plot(raw_values[-sv_len:])
            # pyplot.plot(predictions)
            # pyplot.show()

            return predictions, rmse
        else:
            return predictions, self.RMSE_ABORTED_VALUE

    def future_predict(self, lstm_model):
        # walk-forward validation on the test data
        predictions = list()

        prev_row_value = self.raw_values.iloc[-1]
        prev_value = self.train_scaled[-1, 0:-1]
        for i in range(self.predictions_num):
            if self.m_terminate:
                break

            #TODO
            yhat = self.forecast_lstm(lstm_model, 1, prev_value)
            yhat = self.invert_scale(self.scaler, prev_value, yhat)
            yhat_converted = self.inverse_difference_byprev(prev_row_value, yhat)

            #difference
            values = Series([prev_row_value, yhat_converted])
            diff_values = self.difference(values, 1)
            supervised = self.timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            scaled_values = self.scale(self.scaler, supervised_values)
            prev_row_value = yhat_converted
            prev_value = scaled_values[-1, 0:-1]

            predictions.append(yhat_converted)

        return predictions, self.RMSE_ABORTED_VALUE


class NeuralNetworkTeacher(QtCore.QThread):
    signal_epoch = QtCore.pyqtSignal(int)
    signal_repeat = QtCore.pyqtSignal(int, Predictions)
    signal_complete = QtCore.pyqtSignal()

    def __init__(self, neural_network: NeuralNetwork, parent=None):
        super().__init__(parent)
        self.neural_network = neural_network

    def run(self):
        self.neural_network.make_multi_predictions(lambda i, predictions: self.signal_repeat.emit(i, predictions),
                                                   lambda i: self.signal_epoch.emit(i))
        self.signal_complete.emit()

    def tterminate(self):
        self.neural_network.m_terminate = True
