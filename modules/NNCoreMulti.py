from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from math import sqrt
from numpy import array

from modules.NNCore import INetwork, TimeHistory


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


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


# inverse data transform on forecasts
def inverse_transform(forecasts, scaler):
	inverted = list()

	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# store
		inverted.append(inv_scale)
	return inverted


class NMultiStep(INetwork):
	def __init__(self, data, train_size, repeats, epoch, lag_size, seq_size, batch_size, lstm_neurons, lstm_layer, optimizer):
		super().__init__(data, repeats, epoch, batch_size, lstm_neurons, lstm_layer, optimizer)

		self.train_size = train_size
		self.lag_size = lag_size
		self.seq_size = seq_size
		self.test_scaled = None

		self.prepare_data()

	def make_model(self, batch_size, inp_shape_dim, inp_shape_ddim, output_shape_dim=1):
		model = Sequential()

		for i in range(self.lstm_layers - 1):
			model.add(LSTM(self.lstm_neurons, batch_input_shape=(batch_size, inp_shape_dim, inp_shape_ddim),
							stateful=True, return_sequences=True))
			model.add(Dropout(0.2))

		model.add(LSTM(self.lstm_neurons, batch_input_shape=(batch_size, inp_shape_dim, inp_shape_ddim), stateful=True))
		model.add(Dropout(0.2))
		model.add(Dense(output_shape_dim))

		return model

	def fit_lstm(self, iteration_callback):
		# reshape training into [samples, timesteps, features]
		X, y = self.train_scaled[:, 0:self.lag_size], self.train_scaled[:, self.lag_size:]
		X = X.reshape(X.shape[0], 1, X.shape[1])

		Xt, yt = self.test_scaled[:, 0:self.lag_size], self.test_scaled[:, self.lag_size:]
		Xt = Xt.reshape(Xt.shape[0], 1, Xt.shape[1])

		early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
		train_timer = TimeHistory()
		self.gui_controller.callback_func = iteration_callback

		model = self.make_model(self.batch_size, X.shape[1], X.shape[2], y.shape[1])
		model.compile(loss='mean_squared_error', optimizer=self.optimizer)
		model.fit(X, y, epochs=self.epoch, batch_size=self.batch_size, verbose=0, shuffle=False,
				  callbacks=[train_timer, self.gui_controller, early_stop],
				  validation_data=(Xt, yt))

		# Для разных batch_size при обучении и предсказании
		predict_model = self.make_model(1, X.shape[1], X.shape[2], y.shape[1])

		old_weights = model.get_weights()
		predict_model.set_weights(old_weights)
		predict_model.compile(loss='mean_squared_error', optimizer=self.optimizer)

		return predict_model, train_timer.get_time_delta()

	def prepare_data(self):
		data = array(self.raw_values)
		values = data.reshape(len(data), 1)

		# rescale values to -1, 1
		self.scaler = MinMaxScaler(feature_range=(-1, 1))
		scaled_values = self.scaler.fit_transform(values)
		scaled_values = scaled_values.reshape(len(scaled_values), 1)

		# transform into supervised learning problem X, y
		supervised = series_to_supervised(scaled_values, self.lag_size, self.seq_size)
		supervised_values = supervised.values

		# split into train and test sets
		self.train_scaled, self.test_scaled = supervised_values[:self.train_size], supervised_values[self.train_size:]

	def prediciotns_repeat(self, lstm_model):
		forecasts = list()
		for i in range(len(self.test_scaled)):
			if self.gui_controller.terminate:
				break

			X, y = self.test_scaled[i, 0:self.lag_size], self.test_scaled[i, self.lag_size:]

			# make forecast
			forecast = forecast_lstm(lstm_model, X, self.batch_size)

			# store the forecast
			forecasts.append(forecast)

		if not self.gui_controller.terminate:
			# report performance
			forecasts = inverse_transform(forecasts, self.scaler)
			rmse_values = self.evaluate_forecasts(forecasts)

			return forecasts, rmse_values
		else:
			return forecasts, self.RMSE_ABORTED_VALUE

	# evaluate the RMSE for each forecast time step
	def evaluate_forecasts(self, forecasts):
		rmse_values = list()

		test_pack = [row[self.lag_size:] for row in self.test_scaled]
		test_pack = inverse_transform(test_pack, self.scaler)

		for i in range(self.seq_size):
			actual = [row[i] for row in test_pack]
			predicted = [forecast[i] for forecast in forecasts]
			rmse_values.append(sqrt(mean_squared_error(actual, predicted)))

		return rmse_values
