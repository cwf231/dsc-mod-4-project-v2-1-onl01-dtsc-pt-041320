################################################################
# HELPER FUNCTIONS / CLASSES FOR DISASTER RESPONSE NLP PROJECT #
################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, classification_report,
							 f1_score, recall_score, 
							 precision_score, accuracy_score)
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from nltk.util import ngrams
from nltk.probability import FreqDist

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import kerastuner as kt

import IPython
import os
import json
import joblib


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
        
        top = character * max_len
        mid = f'{character}{" " * left}{string}{" " * right}{character}'
        bot = top
    else:
        # Create modular header boxes depending on the length of the string.
        top = character * (len(f'{string}')+42)
        mid = f'{character}{" " * 20}{string}{" " * 20}{character}'
        bot = top
        
    return f'{top}\n{mid}\n{bot}'


def plot_message_lengths_distribution(series,
									  dist_color,
									  mean_color,
									  std_color, 
									  pop_label='',
									  n_std=3,
									  bins='auto',
									  figsize=(12, 6)):
	"""
	Plot a a population's distribution and 
	a zoomed population around its mean side-by-side.

	Parameters:
	-----------
	series: Pandas Series
		Series to plot.
	dist_color/mean_color/std_color: string
		Color hexcode or color interpretable by matplotlib.
	pop_label: string
		Label to give to the population.
		If none is given, the name of the series will be used.
	n_std: int (default: 3)
		Number of standard deviations to scale to.
	bins: int, None, or 'auto' (default: 'auto')
		Number of bins for the histogram.
		(to be interpreted by seaborn's distplot)
	figsize: tuple (default: (12, 6))
		(width, height) of figure.
	"""
	if not pop_label:
		pop_label = series.name

	# Get mean / STD.
	n_words_mean = series.mean()
	n_words_std = series.std()

	# Plot distributions.
	fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)

	# Full distribution.
	sns.distplot(series, 
	             bins=500,
	             label=f'{pop_label}', 
	             ax=ax1, 
	             color=dist_color)
	ax1.set(title=f'{pop_label} Distribution')

	# Abbreviated distribution.
	sns.distplot(series, 
	             bins=500,
	             label=f'{pop_label}', 
	             ax=ax2, 
	             color=dist_color)
	ax2.set(title=f'{pop_label} Distribution\nZoomed to $\pm{n_std}\sigma$')

	# Zoom
	ax2.set_xlim([
	    n_words_mean - (n_words_std*n_std),
	    n_words_mean + (n_words_std*n_std)
	])

	# Vlines.
	for ax in (ax1, ax2):
	    ax.axvline(n_words_mean, 
	               color=mean_color, 
	               label=f'$\mu$ = {round(n_words_mean, 3)}')
	    ax.axvline(n_words_mean + n_words_std, 
	               color=std_color, 
	               label=f'$\sigma$ = {round(n_words_std, 3)}',
	               ls=':')
	    ax.axvline(n_words_mean - n_words_std, 
	               color=std_color,
	               ls=':')
	ax1.legend()
	ax2.legend()
	fig.tight_layout()
	plt.show()


def plot_wordcloud(wordcloud, figsize=(12, 8)):
	"""
	Plot wordcloud image from WordCloud object.
	"""
	plt.figure(figsize=figsize)
	plt.imshow(wordcloud, interpolation="bilinear")
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


def plot_confusion_matrix(y_true,
                          y_pred,
                          column_name='',
                          figsize=(5,5),
                          normalize_true=True,
                          cmap='Blues',
                          return_fig=False):
	"""
	Plot a confusion matrix from y_true, y_pred.

	Parameters:
	-----------
	y_true: array
		True labels.
	y_pred: array
		Predicted labels (same shape as y_true)
	column_name: string (default: '')
		Title of the plot.
	figsize: tuple (default: (5, 5))
		(width, height) of figure.
	normalize_true: bool (default: True)
		Whether or not the confusion matrix should be normalized
		on the 'true' labels (ie: each row is normalized).
	cmap: str (default: 'Blues')
		Color-map (interpreted by Seaborn)
	return_fig: bool (default: False)
		Whether the function should return the figure as well as showing it.

	Returns:
	--------
	If return_fig is passed as True, the matplotlib figure will be returned.
	"""
	with plt.style.context(['ggplot', 'seaborn-talk']):
		# Get confusion matrix.
		matrix = confusion_matrix(y_true, y_pred)

		# Create figure.
		fig, ax = plt.subplots(figsize=figsize)

		# Plot.
		if normalize_true:
			matrix = normalize(matrix, norm="l1")
			sns.heatmap(matrix, cmap=cmap, 
						vmin=0, vmax=1 if normalize_true else None, 
						annot=True, annot_kws={"size": 15}, 
						ax=ax)
			ax.set(title=column_name, 
				   xlabel='Predicted Label', 
				   ylabel='True Label')

		fig.tight_layout()
		plt.show()

		if return_fig:
			return fig


def fit_thresholds(y_true, 
				   y_pred_raw, 
				   metric, 
				   verbose=True,
				   thresholds=np.arange(0.1, 1, 0.1),
				   return_top_threshold=False):
	"""
	Fit a prediction array to different thresholds.
	The top score (based on a given metric) will be displayed.

	Parameters:
	-----------
	y_true: array
		Array of true labels.
	y_pred_raw: array
		Array of floats - prediction from model.
	metric: string (one of ['f1_score', 'accuracy_score', 'recall', 'precision'])
		The metric to optimize for.
	thresholds: list or array (default: np.arange(0.1, 1, 0.1))
		List or array of values from 0 to 1.0.
		The required threshold to consider a given positive.
	return_top_threshold: bool (default=False)
		Whether to return the top_threshold value (float).
	"""
	# Set up functions dict.
	metrics_dct = {
		'f1_score': f1_score, 
		'accuracy_score': accuracy_score, 
		'recall': recall_score, 
		'precision': precision_score
	}

	if metric not in metrics_dct:
		raise Exception('''
`metric` value must be one of:
	['f1_score', 'accuracy_score', 'recall', 'precision']
''')

	# Set up top lists.
	top_thresh_val = None
	top_score = 0
	top_pred = None

	# Iterate through thresholds and keep top score.
	for threshold in thresholds:
		pred = y_pred_raw.copy()

		# Set predictions based on given threshold.
		pred[pred >= threshold] = 1
		pred[pred < threshold] = 0

		# Determine whether the top score should be replaced.
		score = metrics_dct[metric](y_true, pred)
		if score > top_score:
			top_thresh_val = threshold
			top_score = score
			top_pred = pred

	results = f'Optimal Threshold: {top_thresh_val}. Top Score: {top_score}'
	print(headerize(results))
	if verbose:
		show_scores(y_true, top_pred, header=metric.upper())
	if return_top_threshold:
		return top_thresh_val


def fit_prediction_to_threshold(y_pred, threshold=0.5):
	"""
	Returns a prediction which is interpreted based on a given threshold.
	By default, this will return a binary score where each value is rounded.
	"""
	y = y_pred.copy()
	y[y < threshold] = 0
	y[y >= threshold] = 1
	return y


def evaluate_nn(model, 
				model_history, 
				X_val, 
				y_val, 
				X_test=None, 
				optimize_for='f1_score', 
				header='', 
				column_names=None):
	"""
	...
	"""
	possible = ['f1_score', 'accuracy_score', 'recall', 'precision']
	if optimize_for not in possible:
		raise Exception('''
`optimize_for` value must be one of:
	['f1_score', 'accuracy_score', 'recall', 'precision']
''')
	print(headerize(header))
	plot_history(model_history)

	# Get y_pred.
	y_pred = model.predict(X_val)

	# # Get optimal threshold.
	# thresh = fit_thresholds(y_val, 
	# 						y_pred, 
	# 						optimize_for, 
	# 						verbose=False,
	# 						return_top_threshold=True)
	thresh = 0.5

	# Show scores
	show_scores(y_true=y_val, 
                y_pred=fit_prediction_to_threshold(y_pred, thresh),
                conf_matrix=True,
                header=header,
                column_names=column_names)

	if X_test is not None:
		return fit_prediction_to_threshold(model.predict(X_test), thresh)


def train_dump_model(name, 
					 model, 
					 X_train, 
					 y_train, 
					 X_val, 
					 y_val, 
					 f_path,
					 class_weight=None):
	f_dir = os.path.dirname(f_path)

	# Set callbacks.
	callbacks = [
		EarlyStopping(patience=3),
		# ModelCheckpoint(filepath=checkpoint_path, 
		# 				save_weights_only=True, 
		# 				verbose=1)
	]

	# Train model with checkpoints.
	model_history = model.fit(X_train, 
							  y_train, 
							  epochs=50, 
							  batch_size=32, 
							  validation_data=(X_val, y_val),
							  callbacks=callbacks,
							  class_weight=class_weight,
							  verbose=2)
	# Save.
	model.save(f_path)
	joblib.dump(model_history.history, f_dir+f'/{name}_history.pkl')

	print(headerize(f'{name} - Complete'))
	return model_history


def fit_predict_model(clf,
					  X_train,
					  Y_train,
					  X_val=None,
					  Y_val=None,
					  header='',
					  target_names=None,
					  return_pred=False,
					  target_column_names=None,
					  plot_confusion=True):
	"""
	Fit a given classifier.
	Optional to pass validation data and print f1_score and accuracy score.

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
		on the validation data and show f1_score and accuracy_score.
	Y_val: array (default: None)
		...
	header: string (default: '')
		String to headerize at the top of the report 
		(if validation data is given).
	target_names: list or None (default: None)
		List of strings of the labels in order. 
		(eg [`0`, `1`] --> ['class_negative', 'class_positive'])
	return_pred: bool (default: False)
		Decide whether to return the prediction on the given X_val.
	target_column_names: list of strings or None (default: None)
		Column names for the multi-label confusion matrix.
	plot_confusion: bool (default: True)
		If True, plot_confusion_matrix will be called
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
				conf_matrix=plot_confusion,
				header=header,
				target_names=target_names,
				column_names=target_column_names)

	# Return prediction.
	if return_pred:
		return Y_pred


def show_scores(y_true, 
				y_pred, 
				conf_matrix=False,
				target_names=None,
				header='', 
				column_names=''):
	"""
	Print f1_score & accuracy_score.
	Optional to print recall and precision scores.
	Optional to plot multi_label_conf_matrix for each label.

	Parameters:
	-----------
	y_true: array
		True values.
	y_pred: array
		Predicted values.
	conf_matrix: bool
		If True, plot_confusion_matrix will be called
		and labeled with the given column names.
	target_names: list of strings or None (default: None)
		Names of the targets in order of the labels (ie: [0, 1])
	header: string (default: '')
		String to headerize at the top of the report.
	column_names: string (default: '')
		Column name for the multi-label confusion matrix.
	"""
	if header:
		print(headerize(header))

	dct = classification_report(y_true, y_pred, 
								target_names=target_names,
								output_dict=True)
	if target_names is None:
		target_names = [str(x) for x in sorted(list(set(y_true)))]

	# Scores.
	target_dct = {target: dct[target] for target in target_names}
	print(pd.DataFrame.from_dict(target_dct, orient='index')) # scores per target.
	print()
	print(pd.DataFrame([dct['accuracy']], index=[' '], columns=['accuracy'])) # accuracy
	print()
	avg_dct = {a: dct[a] for a in dct.keys() if 'avg' in a}
	print(pd.DataFrame.from_dict(avg_dct, orient='index')) # micro / micro avg.

	# Confusion matrix.
	if conf_matrix:
		plot_confusion_matrix(y_true,
							  y_pred,
							  column_name=column_names)


def make_scores_df(index_lst=['F1', 'Accuracy', 'Recall', 'Precision']):
	"""
	Create empty dataframe to store model results.
	"""
	df = pd.DataFrame(index=index_lst)
	print(headerize('DataFrame Created Sucessfully'))
	return df


def get_scores(y_true, y_pred):
	"""
	Returns a list of:
		[`f1_score`, `accuracy_score`, `recall_score`, `precision_score`]
	"""
	metrics_lst = [
		f1_score(y_true, y_pred), 
		accuracy_score(y_true, y_pred), 
		recall_score(y_true, y_pred), 
		precision_score(y_true, y_pred)
	]
	return metrics_lst


def plot_history(history, style=['ggplot', 'seaborn-talk']):
	"""
	Plot history from History object (or history dict) 
	once Tensorflow model is trained.

	Parameters:
	-----------
	history:
		History object returned from a model.fit()
	style: string or list of strings (default: ['ggplot', 'seaborn-talk'])
		Style from matplotlib.
	"""
	if type(history) != dict:
		history = history.history

	metrics_lst = [m for m in history.keys() if not m.startswith('val')]
	N = len(metrics_lst)
	with plt.style.context(style):
		fig, ax_lst = plt.subplots(nrows=N, figsize=(8, 4*(N)))
		for metric, ax in zip(metrics_lst, ax_lst.flatten()):
			val_m = f'val_{metric}'
			ax.plot(history[metric], label=metric)
			ax.plot(history[val_m], label=val_m)
			ax.set(title=metric.title(), xlabel='Epoch', ylabel=metric.title())
			ax.legend()
		fig.tight_layout()
		plt.show()


def plot_top_ngrams(N,
					pos_data,
					neg_data,
					pos_color,
					neg_color,
					pos_label='',
					neg_label='',
					header=''):
	"""
	Comparitively plot most common ngrams.

	Parameters:
	-----------
	N: int (must be greater than 1)
		Length of max ngram that is created. 
		Ngrams of range(1, N+1) will be created.
	pos_data / neg_data: list of tokenized data.
		Data to create ngrams.
		One for positive, one for negative.
	pos_color / neg_color: string
		Color to be interpreted by seaborn.
	pos_label / neg_label: string (default: '')
		Title to give to the plot column.
	header: string (default: '')
		Header to print at the start.
	"""
	print(headerize(header))

	# Plot barcharts.
	fig, ax_lst = plt.subplots(nrows=N, ncols=2, figsize=(16, 12))
	for (ax1, ax2), i in zip(ax_lst, range(1, N+1)):
	    if i == 1:
	        ax1.set(title=neg_label)
	        ax2.set(title=pos_label)
	    xneg = []
	    xpos = []
	    yneg = []
	    ypos = []
	    
	    # Append top ngrams for each.
	    for t, count in top_ngrams(neg_data, i, top=5):
	        xneg.append(count)
	        yneg.append(' '.join(t))
	    for t, count in top_ngrams(pos_data, i, top=5):
	        xpos.append(count)
	        ypos.append(' '.join(t))
	        
	    # Plot
	    sns.barplot(x=xneg, y=yneg, orient='h', color=neg_color, ax=ax1)
	    sns.barplot(x=xpos, y=ypos, orient='h', color=pos_color, ax=ax2)
	    ax1.set(xlabel='Frequency')
	    ax2.set(xlabel='Frequency')
	    
	fig.tight_layout()
	plt.show()


def tokenize_series(series_to_fit, 
					series_to_tokenize, 
					num_words=None, 
					pad=True,
					pad_sequences_maxlen=100,
					verbose=True):
	"""
	Fits and transforms a list of text Series on a Tokenizer.
	Returns a fit tokenizer object and the transformed series'.

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
		finished()
		return (tokenizer, padded)
	finished()
	return (tokenizer, tokenized_msg_lst)


def make_embedding_matrix(word_embedder, word_to_index):
	"""
	Create pretrained embedding matrix with dimensions:
		(vocab_len, word_embedder_dimensions)
	Assigns the word embeddings to the appropriate rows associated with 
	the word_to_index.

	Parameters:
	-----------
	word_embedder: dict
		Dictionary with format {word: {array}}.
	word_to_index: dict
		Dictionary with format: {word: idx}.

	Returns: 
		(vocab_len, 
		num_dimensions, 
		matrix)

	**matrix: shape (len(word_to_index) + 1, len({array}))**
	"""
	vocab_len = len(word_to_index) + 1
	num_dimensions = next(iter(word_embedder.values())).shape[0]

	matrix = np.zeros((vocab_len, num_dimensions))
	for word, idx in word_to_index.items():
		if word not in word_embedder:
			continue
		matrix[idx, :] = word_embedder[word]

	return vocab_len, num_dimensions, matrix

def gridsearch_classifiers(classifiers,
						   mean_word_embedder,
						   directory,
						   X_train,
						   Y_train,
						   suffix_label='',
						   cv=3,
						   njobs=-2):
	"""
	GridSearch a list of (classifier_steps, parameters).
	Saves a pickle of the best estimator to the given directory.
	Returns a list of best estimators.

	Parameters:
	-----------
	classifiers: list
		List of tuples in the following format:
			(clf_step, clf_params)
		The `clf_step` should be a step to input to a Pipeline.
			(ex: ('name', clf))
		The `clf_params` should be a dictionary of parameters to try
		through a Pipeline (the key should be '{name}__{parameter}')
		For example:
			>>> classifiers = [
			>>>		(('LOGREG', LogisticRegression()), 
			>>>		 {'LOGREG__C': [0.1, 1., 10.]}),
			>>>		...
			>>>]
	mean_word_embedder: MeanEmbedder Object
		(Instantiated from this file: dis.MeanEmbedder())
	directory: string
		Directory path where to save the models.
	X_train: array
		Data to train the GridSearch on.
	Y_train: array
		Data to train the GridSearch on.
	cv: int (default: 3)
		Number of CrossFolds used for validation.
	njobs: int (default: -2)
		Number of processors to use. 
		If njobs is negative, it will use all but njobs available processors.
	"""
	if not directory.endswith('/'):
		directory += '/'

	save_lst = []
	# Fit rfc, svc, lr models through a GridSearch.
	for clf_step, clf_params in classifiers:
		name = clf_step[0]

		# Build Pipeline & Gridsearch.
		steps = [('MeanWordEmbedder', mean_word_embedder), clf_step]
		pipe = Pipeline(steps=steps, verbose=True)
		clf_grid = GridSearchCV(pipe, 
								clf_params, 
								cv=cv, 
								n_jobs=njobs, 
								verbose=True)

		# Fit & save.
		clf_grid.fit(X_train, Y_train)
		save_lst.append(
			(name, clf_grid.best_estimator_)
		)
		joblib.dump(clf_grid.best_estimator_, 
					f'{directory}{name}{suffix_label}.pkl')
	return save_lst


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
	Initiates with a word vector object.
	Can pass in a dictionary or Word2VecKeyedVectors object

	Transforms an array of words into a mean vector.
	"""
	def __init__(self, word_vectors, verbose=False):
		self.word_vectors = word_vectors

		if type(word_vectors) == dict:
			self.dim = len(word_vectors[list(word_vectors.keys())[0]])
			self.w2v = False
			if verbose:
				print('Loaded from dictionary.')
		elif hasattr(word_vectors, 'vocab'):
			self.dim = len(word_vectors[list(word_vectors.vocab)[0]])
			self.w2v = True
			if verbose:
				print('Loaded from W2V.')
		else:
			raise Exception('`word_vectors` type not recognized.')

		if verbose:
			print(headerize('Success'))
			print(f'Embedding model loaded. Dimensions: {self.dim}')

	def __repr__(self):
		if self.w2v:
			return 'MeanEmbedder(W2V)'
		else:
			return 'MeanEmbedder(GloVe)'

	# Create fit method to cooperate with sklearn Pipelines.
	def fit(self, X, y):
		return self

	def transform(self, X):
		if self.w2v:
			return np.array([
				np.mean(
					[self.word_vectors[w] for w in words 
					 if w in self.word_vectors.vocab]
					or [np.zeros(self.dim)], axis=0) 
				for words in X
				])
		else:
			return np.array([
				np.mean(
					[self.word_vectors[w] for w in words 
					 if w in self.word_vectors]
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

	def __repr__(self):
		self.show_thyself()
		return ''

	def __str__(self):
		return 'DataWarehouse()'

	def show_thyself(self):
		print(headerize('DataWarehouse'))
		self.show_data_shapes()
		print(headerize('Raw Data'))
		self.show_data_shapes(show_processed=False, show_column_desc=True)
		print(headerize('Processed Data'))
		self.show_data_shapes(show_processed=True, show_column_desc=True)
		print(headerize('Columns'))
		print(list(self.processed_train.columns))
		self.show_column_split()

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
								negative_label,
								positive_label,
								title='',
								figsize=(8, 6)):
		"""
		Plot a donut plot showing the distribution of class labels.

		Parameters:
		-----------
		dataset: string (one of ['train', 'val', 'test'])
			Dataset to pass. 
			The internal processed corresponding DataFrame will be used.
		positive_color: string
			Bar color for positive values.
		negative_color: string
			Bar color for negative values.
		negative_label: string
			Label for negative values on the donut.
		positive_label: string
			Label for positive values on the donut.
		title: string (default: '')
			Header for the top of the printout.
		figsize: tuple (default: (8, 6))
			Size in inches for the plot.
		"""
		if type(self.target_columns) == list and len(self.target_columns) > 1:
			raise ValueError('`predictive_columns` must be a single value.')

		print(headerize(title))
		trgt_dct = {
			'train': self.processed_train, 
			'val': self.processed_val, 
			'test': self.processed_test
		}

		if type(self.target_columns) == list:
			col = self.target_columns[0]
		else:
			col = self.target_columns
		trgt = trgt_dct[dataset][col]

		# Plot donut chart.
		fig, ax = plt.subplots(figsize=figsize)
		ax.pie(trgt.value_counts(), 
		       colors=[negative_color, positive_color],
		       labels=[negative_label, positive_label],
		       autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p*len(trgt)/100),
		       textprops={'fontsize': 18})
		ax.add_artist(plt.Circle((0,0), 0.5, color='white'))
		ax.set(title=f'{col.title()} Distribution')
		fig.tight_layout()


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
