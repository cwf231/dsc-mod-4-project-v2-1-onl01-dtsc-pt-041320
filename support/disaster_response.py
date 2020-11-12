################################################################
# HELPER FUNCTIONS / CLASSES FOR DISASTER RESPONSE NLP PROJECT #
################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (multilabel_confusion_matrix, 
							 f1_score, recall_score, 
							 precision_score, hamming_loss)
from sklearn.preprocessing import normalize

from nltk.util import ngrams
from nltk.probability import FreqDist

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import kerastuner as kt

import IPython
import os
import json


def check_for_glove(default_dir='./models/', fname='glove.6B.100d.txt'):
	if fname in os.listdir(default_dir):
		print(headerize('SUCCESS - GloVe Model Found'))
	else:
		print(headerize('WARNING - Failed to load GloVe Model'))
		raise Exception(f'''
This notebook cannot be run properly without the 
GloVe model ``{fname}``.

This model can be downloaded from:
	> http://nlp.stanford.edu/data/glove.6B.zip
and saved into the directory ``{default_dir}``.''')


def underline(string, character='-'):
    """
    Return a string of a given character with the length of a given string.
    """
    return character * len(string)
    
    
def headerize(string, character='*', max_len=80):
    """
    Return a given string with a box (of given character) around it.
    """
    if max_len:
        # Create uniform size boxes for headers with centered text.
        if len(string) > max_len-2:
            string = string[:max_len-5] + '...'
            
        total_space = max_len - 2 - len(string)
        left = total_space // 2
        if total_space % 2 == 0:
            right = left
        else:
            right = left + 1
        
        top = character * 80
        mid = f'{character}{" " * left}{string}{" " * right}{character}'
        bot = top
    else:
        # Create modular header boxes depending on the length of the string.
        top = character * (len(f'{string}')+42)
        mid = f'{character}{" " * 20}{string}{" " * 20}{character}'
        bot = top
        
    return f'{top}\n{mid}\n{bot}'


def plot_wordcloud(wordcloud, figsize=(12, 8)):
	"""
	Plot wordcloud image from WordCloud object.
	"""
	plt.figure(figsize=figsize)
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()


def top_ngrams(tokenized_words, n=2, top=10):
	"""
	Returns a list of tuples of the `top` ngrams of length n.

	`tokenized_words` should be a Series.values of tokenized words.
	"""
	all_ngrams = []
	for each in tokenized_words:
		all_ngrams += ngrams(each, n)
	return FreqDist(all_ngrams).most_common(top)


def plot_multi_label_confusion_matrix(y_true, 
                                      y_pred,
                                      column_labels=None,
                                      normalize_true=True,
                                      dimensions_of_square=4,
                                      cmap='Blues',
                                      return_fig=False):
    """
    Create a One-vs-Rest confusion matrix 
    for each label in a multi-label classification.


    Parameters:
    -----------
    y_true: array
        True labels.
    y_pred: array
        Predicted labels (same shape as y_true)
    column_labels: list or None (default: None)
        List of label names to title each plot.
        If ``None``, each plot will be titled with an integer.
    normalize_true: bool (default: True)
        Whether or not the confusion matrix should be normalized
        on the 'true' labels (ie: each row is normalized).
    dimensions_of_square: int (default: 5)
        Width of each square in the subplot. 
        (Height will be ``dimensions_of_square - 1`` to maintain ratio.)
    cmap: str (default: 'Blues')
        Color-map (interpreted by Seaborn)
    return_fig: bool (default: False)
        Whether the function should return the figure as well as showing it.

    Returns:
    --------
    If return_fig is passed as True, the matplotlib figure will be returned.
    """
    with plt.style.context(['ggplot', 'seaborn-talk']):
    	# Define helper function.
        round_up = lambda numerator, denominator: -(-numerator // denominator)

        # Get multi-label confusion matrix - OvR for each label.
        multi_conf_matrix = multilabel_confusion_matrix(y_true, 
                                                        y_pred)
        num_labels = len(multi_conf_matrix)

        # Setup figure specs.
        MAX_ROWS = 4
        MAX_WIDTH = 12
        if num_labels > MAX_ROWS:
            ncols = MAX_ROWS
        else:
            ncols = num_labels
        nrows = round_up(num_labels, MAX_ROWS)

        w = MAX_WIDTH if ncols * dimensions_of_square > MAX_WIDTH \
            else ncols * dimensions_of_square
        h = nrows * (dimensions_of_square-1)
        figsize = (w, h)

        # Create figure.
        fig, ax_lst = plt.subplots(ncols=ncols, 
                                   nrows=nrows,
                                   figsize=figsize)
        ax_lst = ax_lst.flatten()

        if column_labels is None:
            column_labels = range(num_labels)

        # Plot.
        for ax, matrix, label in zip(ax_lst, 
        							 multi_conf_matrix, 
        							 column_labels):
            if normalize_true:
                matrix = normalize(matrix, norm="l1")
            sns.heatmap(matrix, cmap=cmap, 
                        vmin=0, vmax=1 if normalize_true else None, 
                        annot=True, annot_kws={"size": 15}, 
                        ax=ax)
            ax.set(title=label, 
            	   xlabel='Predicted Label', 
            	   ylabel='True Label')

        # Remove unused axes in subplot.
        ax_to_del = ax_lst[num_labels:]
        for ax in ax_to_del:
            fig.delaxes(ax)

        fig.tight_layout()
        plt.show()

        if return_fig:
            return fig


def fit_thresholds(y_true, 
				   y_pred_raw, 
				   metric, 
				   thresholds=np.arange(0.1, 1, 0.1),
				   return_val_score_pred=False):
	"""
	Fit a prediction array to different thresholds.
	The top score (based on a given metric) will be displayed.

	Parameters:
	-----------
	y_true: array
		Array of true labels.
	y_pred_raw: array
		Array of floats - prediction from model.
	metric: string (one of ['f1_score', 'hamming_loss', 'recall', 'precision'])
		The metric to optimize for.
	thresholds: list or array (default: np.arange(0.1, 1, 0.1))
		List or array of values from 0 to 1.0.
		The required threshold to consider a given positive.
	return_val_score_pred: bool (default=False)
		Whether to return the values in addition to printing the report.
	"""
	# Set up functions dict.
	metrics_dct = {
		'f1_score': f1_score, 
		'hamming_loss': hamming_loss, 
		'recall': recall_score, 
		'precision': precision_score
	}

	if metric not in metrics_dct:
		raise Exception('''
`metric` value must be one of:
	['f1_score', 'hamming_loss', 'recall', 'precision']
''')

	# Set up top lists.
	top_thresh_val, top_score, top_pred = (None, 0, None)
	if metric == 'hamming_loss':
		top_score = 1

	# Iterate through thresholds and keep top score.
	for threshold in thresholds:
		pred = y_pred_raw.copy()

		# Set predictions based on given threshold.
		pred[pred >= threshold] = 1
		pred[pred < threshold] = 0

		# Determine whether the top score should be replaced.
		replace = False
		if metric == 'hamming_loss':
			score = metrics_dct[metric](y_true, pred)
			if score < top_score:
				replace = True
		else:
			score = metrics_dct[metric](y_true, pred, average='samples')
			if score > top_score:
				replace = True

		if replace:
			top_thresh_val, top_score, top_pred = (threshold, score, pred)

	results = f'Optimal Threshold: {top_thresh_val}. Top Score: {top_score}'
	print(headerize(results))
	show_scores(y_true, top_pred, header=metric.upper())


def fit_predict_model(clf,
					  X_train,
					  Y_train,
					  X_val=None,
					  Y_val=None,
					  header='',
					  return_pred=False,
					  target_column_names=None,
					  plot_confusion=True):
	"""
	Fit a given classifier.
	Optional to pass validation data and print f1_score and hamming loss.

	Parameters:
	-----------
	clf: model or pipeline
		Classifier to fit.
	X_train: array
		X training data
	Y_train: array
		y training data
	X_val: array or None (default: None)
		If passed with Y_val, the model will predict 
		on the validation data and show f1_score and hamming_loss.
	Y_val: array (default: None)
		...
	header: string (default: '')
		String to headerize at the top of the report 
		(if validation data is given).
	return_pred: bool (default: False)
		Decide whether to return the prediction on the given X_val.
	target_column_names: list of strings or None (default: None)
		Column names for the multi-label confusion matrix.
	plot_confusion: bool (default: True)
		If True, plot_multi_label_confusion_matrix will be called
		and labeled with the given column names.
	"""
	# Fit.
	clf.fit(X_train, Y_train)
	if X_val is None or Y_val is None:
		return

	# Predict.
	Y_pred = clf.predict(X_val)

	# Show scores.
	show_scores(Y_val, 
				Y_pred, 
				multi_label_conf_matrix=plot_confusion,
				header=header,
				column_names=target_column_names)

	# Return prediction.
	if return_pred:
		return Y_pred


def show_scores(y_true, 
				y_pred, 
				multi_label_conf_matrix=False,
				recall=True,
				precision=True,
				header='', 
				column_names=None):
	"""
	Print f1_score & hamming_loss.
	Optional to print sample-average recall and precision scores.
	Optional to plot multi_label_conf_matrix for each label.

	Parameters:
	-----------
	y_true: array
		True values.
	y_pred: array
		Predicted values.
	recall: bool (default: True)
		Print recall_score (average='samples')
	precision: bool (default: True)
		Print precision_score (average='samples')
	multi_label_conf_matrix: bool
		If True, plot_multi_label_confusion_matrix will be called
		and labeled with the given column names.
	header: string (default: '')
		String to headerize at the top of the report.
	column_names: list of strings or None (default: None)
		Column names for the multi-label confusion matrix.
	"""
	if header:
		print(headerize(header))

	# Scores.
	print('F1 Score:       ', f1_score(y_true, y_pred, 
									   average='samples'))
	print('Hamming Loss:   ', hamming_loss(y_true, y_pred))
	if recall:
		print('Recall Score:   ', recall_score(y_true, y_pred, 
											   average='samples'))
	if precision:
		print('Precision Score:', precision_score(y_true, y_pred, 
												  average='samples'))

	# Multi-label_confusion matrix.
	if multi_label_conf_matrix:
		plot_multi_label_confusion_matrix(y_true,
										  y_pred,
										  column_labels=column_names)


def plot_history(history,
				 plots,
				 figsize=(12, 8),
				 style=['ggplot', 'seaborn-talk']):
	"""
	Plot history from History object once Tensorflow model is trained.

	Parameters:
	-----------
	history:
		History object returned from a model.fit()
	plots: string or list of strings
		What is being plotted from the history object.
		Should be keys in history.history.
	figsize: tuple (default: (12, 8))
		(width, height) of figure.
	style: string or list of strings (default: ['ggplot', 'seaborn-talk'])
		Style from matplotlib.
	"""
	history = history.history
	if len(history) == 2 * len(plots):
		val = True
	plot1, plot2 = plots
	with plt.style.context(style):
		fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
		ax1.plot(history[plot1], label=plot1)
		ax2.plot(history[plot2], label=plot2)
		if val:
			ax1.plot(history[f'val_{plot1}'], label=f'val_{plot1}')
			ax2.plot(history[f'val_{plot2}'], label=f'val_{plot2}')
		ax1.set(title=plot1, xlabel='Epoch', ylabel=plot1)
		ax2.set(title=plot2, xlabel='Epoch', ylabel=plot2)
		for ax in (ax1, ax2):
			ax.legend()
		fig.tight_layout()
		plt.show()


def tokenize_series(series_to_fit, 
					series_to_tokenize, 
					num_words=None, 
					pad=True,
					pad_sequences_maxlen=100,
					return_tokenizer=True,
					verbose=True):
	"""
	Fits and transforms a list of text Series on a Tokenizer.

	Parameters:
	-----------
	series_to_fit: pandas Series (or list-like)
		Series to fit to the Tokenizer.
	series_to_tokenize: list of pandas Series (or list-like)
		List of Series to tokenize.
	num_words: int (default: None)
		The maximum number of words to keep, based
        on word frequency. Only the most common `num_words-1` words will
        be kept.
	pad: bool (default: True)
		Whether to pad requences before returning them.
	pad_sequences_maxlen: int (default: 100)
		Max length of the sequences when padding.
	return_tokenizer: bool (default: True)
		Whether to return the tokenizer as well as the tokenized Series'.
	verbose: bool (default: True)
		Whether to print examples of the process.
	"""
	def finished():
		print(headerize('Finished'))

	# Fit tokenizer to text.
	tokenizer = Tokenizer(num_words=num_words)
	tokenizer.fit_on_texts(list(series_to_fit))
	if verbose:
		print(headerize('Tokenizer Created & Fit'))
		print('Ex:')
		display(list(series_to_fit)[:2])

	# List of tokenized messages.
	tokenized_msg_lst = [tokenizer.texts_to_sequences(list(series))
						 for series in series_to_tokenize]
	if verbose:
		print(headerize("Series' Tokenized"))
		print('Ex:')
		display(tokenized_msg_lst[0][:2])

	# Pad sequences to regularized shape.
	if pad:
		padded = [pad_sequences(msg, maxlen=pad_sequences_maxlen)
				  for msg in tokenized_msg_lst]
		if verbose:
			print(headerize('Tokenized'))
			print('Ex:')
			display(padded[0][:2])
		if return_tokenizer:
			finished()
			return (tokenizer, padded)
		else:
			finished()
			return padded
	if return_tokenizer:
		finished()
		return (tokenizer, tokenized_msg_lst)
	else:
		finished()
		return tokenized_msg_lst


def build_model(num_words, 
				embedding_size, 
				gru=False, 
				lstm=False,
				gru_units=128,
				lstm_units=128,
				dense_units=50,
				learning_rate=0.001
				):
	"""
	Build and return Keras model using GRU or LSTM and a preset architecture.
	Add one of 'gru' or 'lstm'.
	"""
	if gru is False and lstm is False:
		raise Exception('Must select `gru` or `lstm`.')

	model = models.Sequential()

	model.add(layers.Embedding(num_words, embedding_size))
	if gru:
		model.add(
			layers.Bidirectional(
				layers.GRU(gru_units, return_sequences=True)
				)
			)
	if lstm:
		model.add(
			layers.Bidirectional(
				layers.LSTM(lstm_units, return_sequences=True)
				)
			)
	model.add(layers.GlobalMaxPool1D())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(dense_units, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(24, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer=optimizers.Adam(learning_rate=learning_rate),
				  metrics=[tf.keras.metrics.Recall()])
	return model


def keras_tuner_hyperband(X_train, 
						  y_train, 
						  X_val, 
						  y_val, 
						  directory, 
						  project_name='disaster_response'):
	def build_tuner_model(hp):
		model = models.Sequential()
		model.add(layers.Embedding(20_000, 128))

		# Gridsearch GRU units.
		gru_units = hp.Int('gru_units', 
						   min_value=64, 
						   max_value=512, 
						   step=32)
		model.add(layers.GRU(units=gru_units, return_sequences=True))

		model.add(layers.GlobalMaxPool1D())
		model.add(layers.Dropout(0.5))

		# Gridsearch number of dense units (Dense layer 1).
		dense_units1 = hp.Int('dense_units1', 
							  min_value=64, 
							  max_value=512, 
							  step=32)
		model.add(layers.Dense(units=dense_units1, activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(24, activation='sigmoid'))

		# Gridsearch learning_rate
		hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 
															  0.0001])
		model.compile(loss='binary_crossentropy',
					  optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
					  metrics=['accuracy'])
		return model

	# Instantiate tuner with above function.
	tuner = kt.Hyperband(build_tuner_model,
						 objective='val_accuracy', 
						 max_epochs=10,
						 factor=3,
						 directory=directory,
						 project_name=project_name)
	tuner.search(X_train, 
				 y_train, 
				 epochs=10, 
				 validation_data=(X_val, y_val), 
				 callbacks=[ClearTrainingOutput()])

	print(headerize('Tuning Complete'))
	best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
	opt_gru = best_hps.get('gru_units')
	opt_dense1 = best_hps.get('dense_units1')
	learning_rate = best_hps.get('learning_rate')
	print('Optimal Settings:')
	print(f'\tGRU UNITS           : {opt_gru}')
	print(f'\tDENSE UNITS (LAYER): {opt_dense1}')
	print(f'\tLEARNING RATE       : {learning_rate}')

	return tuner


class MeanEmbedder:
	"""
	Mean Word Embedder.
	Initiates with a GloVe dictionary.

	Transforms an array of words into a mean vector.
	"""
	def __init__(self, glove, verbose=False):
		self.glove = glove
		self.dim = len(self.glove[list(self.glove.keys())[0]])
		if verbose:
			print(headerize('Success'))
			print(f'Glove model loaded. Dimensions: {self.dim}')

	def __repr__(self):
		return 'MeanEmbedder()'

	# Create fit method to cooperate with sklearn Pipelines.
	def fit(self, X, y):
		return self

	def transform(self, X):
		return np.array([
			np.mean(
				[self.glove[w] for w in words if w in self.glove]
				or [np.zeros(self.dim)], axis=0) 
			for words in X
			])


class DataWarehouse:
	"""
	A data-handling class - designed for data from:
		https://appen.com/datasets/combined-disaster-response-data/
	Loads train / val / test data from a directory.
	"""
	def __init__(self, 
				 directory='./data/',
				 train_fname='disaster_response_messages_training.csv',
				 val_fname='disaster_response_messages_validation.csv',
				 test_fname='disaster_response_messages_test.csv',
				 col_desc_fname='column_descriptions.csv',
				 verbose=True):
		"""
		Load in data from given directory and filenames.
		**Data is expected raw from the combined-disaster-response-data 
		dataset on appen.com**
		"""
		d_type_dct = {
			'id': 'int64',
			'split': 'object',
			'message': 'object',
			'original': 'object',
			'genre': 'object',
			'related': 'int64',
			'PII': 'int64',
			'request': 'int64',
			'offer': 'int64',
			'aid_related': 'int64',
			'medical_help': 'int64',
			'medical_products': 'int64',
			'search_and_rescue': 'int64',
			'security': 'int64',
			'military': 'int64',
			'child_alone': 'int64',
			'water': 'int64',
			'food': 'int64',
			'shelter': 'int64',
			'clothing': 'int64',
			'money': 'int64',
			'missing_people': 'int64',
			'refugees': 'int64',
			'death': 'int64',
			'other_aid': 'int64',
			'infrastructure_related': 'int64',
			'transport': 'int64',
			'buildings': 'int64',
			'electricity': 'int64',
			'tools': 'int64',
			'hospitals': 'int64',
			'shops': 'int64',
			'aid_centers': 'int64',
			'other_infrastructure': 'int64',
			'weather_related': 'int64',
			'floods': 'int64',
			'storm': 'int64',
			'fire': 'int64',
			'earthquake': 'int64',
			'cold': 'int64',
			'other_weather': 'int64',
			'direct_report': 'int64'
		}
		self.train = pd.read_csv(directory + train_fname, dtype=d_type_dct)
		self.val = pd.read_csv(directory + val_fname, dtype=d_type_dct)
		self.test = pd.read_csv(directory + test_fname, dtype=d_type_dct)
		self.column_descriptions = pd.read_csv(directory + col_desc_fname, 
											   index_col='Column', 
											   dtype=d_type_dct)

		self.target_columns = list(self.train.columns[5:])
		self.predictive_columns = [c for c in self.train.columns 
								   if c not in self.target_columns]

		self.processed_train = self.train.copy()
		self.processed_val = self.val.copy()
		self.processed_test = self.test.copy()

		self.processed_data = [
			self.processed_train,
			self.processed_val,
			self.processed_test
		]

		if verbose:
			sucess_msg = 'Data loaded sucessfully.'
			print(headerize(sucess_msg))
			self.show_data_shapes(show_processed=False, show_column_desc=True)

	def show_column_split(self):
		"""
		Print out the DataWarehouse's predictive columns (used for X)
		and its target columns (used for Y).
		"""
		print(headerize('Column Split'))
		print('Predictive Columns (X):\n\t', self.predictive_columns)
		print()
		print('Target Columns (Y):\n\t', self.target_columns)

	def show_data_shapes(self, show_processed=True, show_column_desc=False):
		"""
		Print shape of training, val, and test data.
		Choose either the processed data (default) or the raw data.
		"""
		print(headerize('Data Shapes'))
		if show_processed:
			print(f'Processed Training Data:\n\t{self.processed_train.shape}')
			print(f'Processed Val Data:\n\t{self.processed_val.shape}')
			print(f'Processed Test Data:\n\t{self.processed_test.shape}')
		else:
			print(f'Training Data:\n\t{self.processed_train.shape}')
			print(f'Val Data:\n\t{self.processed_val.shape}')
			print(f'Test Data:\n\t{self.processed_test.shape}')
		if show_column_desc:
			print()
			print(f'Column Descriptions:\n\t{self.column_descriptions.shape}')

	def plot_label_distribution(self, 
								dataset, 
								positive_color,
								negative_color,
								normalize=False,
								figsize=(12, 12)):
		"""
		Plot a stacked horizontal bar chart 
		showing the distribution of class labels.

		Parameters:
		-----------
		dataset: string (one of ['train', 'val', 'test', 'train_val'])
			Dataset to pass. The internal processed 
				corresponding DataFrame will be used.
			If ``train_val`` is passed, a figure 
				with both columns will be shown.
		positive_color: string
			Bar color for positive values.
		negative_color: string
			Bar color for negative values.
		normalize: bool (default: False)
			Whether the bars should should parts of the total
			or percentage of 1.0.
		figsize: tuple (default: (12, 12)
			Size in inches for the plot.
		"""
		if dataset != 'train_val':
			data_map = {
				'train': self.processed_train[self.target_columns],
				'val': self.processed_val[self.target_columns],
				'test': self.processed_test[self.target_columns]
			}
			df = data_map[dataset].copy()

			# Set up values.
			num_positive = df.sum()
			num_positive.sort_values(inplace=True)
			if normalize:
				num_positive /= len(df)
				num_negative = (1 - num_positive) * -1
			else:
				num_negative = (len(df) - num_positive) * -1
			labels = num_positive.index

			# Plot values.
			fig, ax = plt.subplots(figsize=figsize)
			ax.barh(labels, num_negative, 
					label='Does Not Have Label', 
					color=negative_color)
			ax.barh(labels, num_positive, 
					label='Has Label', 
					color=positive_color, 
					left=0)
			ax.set(title='Distribution of Labels',
			       ylabel='Label',
			       xlabel='Count')
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
			          fancybox=True, shadow=True, ncol=5)
			if normalize:
				ax.set_xlim(-1, 1)
			else:
				ax.set_xlim(len(df)*-1, len(df))
			fig.tight_layout()
			plt.show()
		else:
			df1 = self.processed_train[self.target_columns]
			df2 = self.processed_val[self.target_columns]

			# Set up values.
			num_positive1 = df1.sum()
			num_positive1.sort_values(inplace=True)
			if normalize:
				num_positive1 /= len(df1)
				num_negative1 = (1 - num_positive1) * -1
			else:
				num_negative1 = (len(df1) - num_positive1) * -1
			labels = num_positive1.index

			num_positive2 = df2.sum()[labels]
			if normalize:
				num_positive2 /= len(df2)
				num_negative2 = (1 - num_positive2) * -1
			else:
				num_negative2 = (len(df2) - num_positive2) * -1

			# Plot values.
			fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)

			# ax1
			ax1.barh(labels, num_negative1, 
					 label='Does Not Have Label', 
					 color=negative_color)
			ax1.barh(labels, num_positive1, 
					 label='Has Label', 
					 color=positive_color, 
					 left=0)
			ax1.set(title='Distribution of Labels - Train',
			        ylabel='Label',
			        xlabel='Count')
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
			           fancybox=True, shadow=True, ncol=5)

			# ax2
			ax2.barh(labels, num_negative2, 
					 label='Does Not Have Label', 
					 color=negative_color)
			ax2.barh(labels, num_positive2, 
					 label='Has Label', 
					 color=positive_color, 
					 left=0)
			ax2.set(title='Distribution of Labels - Validation',
			        ylabel='Label',
			        xlabel='Count')
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
			           fancybox=True, shadow=True, ncol=5)

			if normalize:
				ax1.set_xlim(-1, 1)
				ax2.set_xlim(-1, 1)
			else:
				ax1.set_xlim(len(df1)*-1, len(df1))
				ax2.set_xlim(len(df2)*-1, len(df2))

			fig.tight_layout()
			plt.show()

	def drop_column(self, column):
		"""
		Drops a given column from all dataframes.
		Replaces current processed_ dataframes.
		
		Parameters:
		-----------
		column: string (or list of strings)
			column names to drop.
		"""
		for df in self.processed_data:
			df.drop(column, axis=1, inplace=True)

		# Clean up target_columns and predictive_columns.
		if type(column) == str:
			column = list(column)
		for c in column:
			if c in self.predictive_columns:
				self.predictive_columns.remove(c)
			elif c in self.target_columns:
				self.target_columns.remove(c)

		print(headerize('Success'))
		print('Columns dropped:\n\t', column)
		print()
		self.show_data_shapes()

	def map_to_all(self, column, funct):
		"""
		Apply a map function to [column] in all processed_ dataframes.
		"""
		for df in self.processed_data:
			df[column] = df[column].map(funct)
		print(headerize('Success'))

	def X_train(self, column=None):
		"""
		Returns a column or column list from 
		the DataWarehouse's processed_train.
		If no column is passed, the columns from self.predictive_columns
		will be used.

		Parameters:
		-----------
		column: string or list of strings (default: None)
			The column names that represent X.
		"""
		if column is None:
			column = self.predictive_columns
		return self.processed_train[column].values

	def Y_train(self, column=None):
		"""
		Returns a column or column list from 
		the DataWarehouse's processed_train.
		If no column is passed, the columns from self.target_columns
		will be used.

		Parameters:
		-----------
		column: string or list of strings (default: None)
			The column names that represent Y.
		"""
		if column is None:
			column = self.target_columns
		return self.processed_train[column].values

	def X_val(self, column=None):
		"""
		Returns a column or column list from 
		the DataWarehouse's processed_val.
		If no column is passed, the columns from self.predictive_columns
		will be used.

		Parameters:
		-----------
		column: string or list of strings (default: None)
			The column names that represent X.
		"""
		if column is None:
			column = self.predictive_columns
		return self.processed_val[column].values

	def Y_val(self, column=None):
		"""
		Returns a column or column list from 
		the DataWarehouse's processed_val.
		If no column is passed, the columns from self.target_columns
		will be used.

		Parameters:
		-----------
		column: string or list of strings (default: None)
			The column names that represent X.
		"""
		if column is None:
			column = self.target_columns
		return self.processed_val[column].values

	def X_test(self, column=None):
		"""
		Returns a column or column list from 
		the DataWarehouse's processed_test.
		If no column is passed, the columns from self.predictive_columns
		will be used.

		Parameters:
		-----------
		column: string or list of strings (default: None)
			The column names that represent X.
		"""
		if column is None:
			column = self.predictive_columns
		return self.processed_test[column].values

	def Y_test(self, column=None):
		"""
		Returns a column or column list from 
		the DataWarehouse's processed_test.
		If no column is passed, the columns from self.target_columns
		will be used.

		Parameters:
		-----------
		column: string or list of strings (default: None)
			The column names that represent X.
		"""
		if column is None:
			column = self.target_columns
		return self.processed_test[column].values


class ClearTrainingOutput(tf.keras.callbacks.Callback):
	def on_train_end(*args, **kwargs):
		IPython.display.clear_output(wait=True)
