from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array

from modules.NNCore import INetwork

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	print('fc_len: ', len(forecasts))
	print('series_len: ', len(series))
	print('n_test: ', n_test)

	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

class NMultiStep(INetwork):
	def __init__(self, data, train_size, repeats, epoch, lag_size, seq_size, batch_size, lstm_neurons):
		super().__init__(data, repeats, epoch, batch_size, lstm_neurons)

		self.train_size = train_size
		self.lag_size = lag_size
		self.seq_size = seq_size
		self.test_scaled = None

		self.prepare_data()

	def fit_lstm(self, iteration_callback):
		# reshape training into [samples, timesteps, features]
		X, y = self.train_scaled[:, 0:self.lag_size], self.train_scaled[:, self.lag_size:]
		X = X.reshape(X.shape[0], 1, X.shape[1])
		# design network
		model = Sequential()
		model.add(LSTM(self.lstm_neurons, batch_input_shape=(self.batch_size, X.shape[1], X.shape[2]), stateful=True))
		model.add(Dense(y.shape[1]))
		model.compile(loss='mean_squared_error', optimizer='adam')
		# fit network
		for i in range(self.epoch):
			if self.m_terminate:
				break

			model.fit(X, y, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=False)
			model.reset_states()

			if iteration_callback:
				iteration_callback(i)
		return model

	def prepare_data(self):
		# transform data to be stationary
		diff_series = difference(self.raw_values, 1)
		diff_values = diff_series.values
		diff_values = diff_values.reshape(len(diff_values), 1)

		# rescale values to -1, 1
		self.scaler = MinMaxScaler(feature_range=(-1, 1))
		scaled_values = self.scaler.fit_transform(diff_values)
		scaled_values = scaled_values.reshape(len(scaled_values), 1)

		scaled_values = self.raw_values

		# transform into supervised learning problem X, y
		supervised = series_to_supervised(scaled_values, self.lag_size, self.seq_size)
		supervised_values = supervised.values

		# split into train and test sets
		self.train_scaled, self.test_scaled = supervised_values[:self.train_size], supervised_values[self.train_size:]

	def prediciotns_repeat(self, lstm_model):
		forecasts = list()
		for i in range(len(self.test_scaled)):
			if self.m_terminate:
				break

			X, y = self.test_scaled[i, 0:self.lag_size], self.test_scaled[i, self.lag_size:]

			# make forecast
			forecast = forecast_lstm(lstm_model, X, self.batch_size)

			# store the forecast
			forecasts.append(forecast)

		if not self.m_terminate:
			# report performance
			forecasts = inverse_transform(self.raw_values, forecasts, self.scaler, len(self.test_scaled) + 2)
			rmse = self.evaluate_forecasts(forecasts)

			# plot_forecasts(self.raw_values, forecasts, len(self.test_scaled))

			return forecasts, rmse
		else:
			return forecasts, self.RMSE_ABORTED_VALUE

	# evaluate the RMSE for each forecast time step
	def evaluate_forecasts(self, forecasts):
		test_pack = [row[self.lag_size:] for row in self.test_scaled]
		test_pack = inverse_transform(self.raw_values, test_pack, self.scaler, len(self.test_scaled) + 2)

		for i in range(self.seq_size):
			actual = [row[i] for row in test_pack]
			predicted = [forecast[i] for forecast in forecasts]
			rmse = sqrt(mean_squared_error(actual, predicted))
			print('RMSE: ', rmse)

		return 1
