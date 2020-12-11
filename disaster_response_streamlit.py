import streamlit as st
import os
import joblib
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Prevents using GPU.
TOKENIZER_PATH = './models/tokenizer/tokenizer.pkl'
MODEL_PATH = './models/tensorflow/rnn_glove.h5'


@st.cache(allow_output_mutation=True)
def load(tokenizer_path, model_path):
	tokenizer = joblib.load(TOKENIZER_PATH)
	model = load_model(MODEL_PATH)
	return tokenizer, model


def prepare_text_for_model(text, tokenizer, max_length=65):
    """
    Transforms text data to a tokenized/padded version suitable for the model.
    
    Parameters:
    -----------
    text: string to analyze.
    tokenizer: Tokenizer object (pre-fit).
    
    Returns:
    --------
    Padded / Tokenized version of data.
    """
    tokenized_data = tokenizer.texts_to_sequences([text])
    padded_data = sequence.pad_sequences(tokenized_data, maxlen=max_length)
    return padded_data


def predict_with_model(processed_data, model):
    """
    Predict the processed data with the model.
    
    Returns (aid_related, y_pred)
    """
    y_pred = model.predict(processed_data)[0, 0]
    aid_related = int(y_pred >= 0.5)
    return aid_related, y_pred


def inference(text, tokenizer, model):
    """Returns a verbose prediction of a given string."""
    processed_data = prepare_text_for_model(text, tokenizer)
    aid_related, y_pred = predict_with_model(processed_data, model)
    likelihood = f'(Likelihood aid-related: {round(y_pred*100, 3)}%)'
    if aid_related:
        return f'This message is aid-related. {likelihood}'
    return f'This message is not aid-related. {likelihood}'


def main():
	st.title('Disaster Response - Message Identification')
	st.write('This app is designed to show predictions on whether a given text is likely to be `aid-related` or not. A model like this can be used to quickly label a high volume of texts during times when it is important to find messages that are labeled as important.')
	st.write('The model being used is a Recurrent Neural Network build with Tensorflow. It uses GloVe weight embeddings and trained for only 8 epochs.')
	st.write('On unseen text data, 80.67% of `aid-related` messages were found, 80.74% of `aid-related` predictions were correct. The model scored an 82.69% overall accuracy.')
	st.image(Image.open('./images/neuron.png'), use_column_width=True)
	st.write('Please fill in some text into the left sidebar, then press the button below. (The messages can be any length)')

	text = st.sidebar.text_input('Message:', 'We are out of food and water.')
	st.write('Current text:')
	st.write('   ', text)

	if (st.button('Click here for results.')):
		tokenizer, model = load(TOKENIZER_PATH, MODEL_PATH)
		result = inference(text, tokenizer, model)
		st.write(result)


if __name__ == '__main__':
	main()
