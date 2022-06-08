import tensorflow as tf
from tensorflow.keras import backend as K
from jiwer import wer


def predict(model, audio, tokenizer, int_to_char, actual=None):
    pred_audios = tf.convert_to_tensor([audio])
    y_pred = model.predict(pred_audios)
    input_shape = tf.keras.backend.shape(y_pred)
    input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(input_shape[1], 'float32')
    prediction = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=False)[0][0]
    pred = K.eval(prediction).flatten().tolist()
    pred = [i for i in pred if i != -1]

    predicted_text = tokenizer.decode_text(pred, int_to_char)

    error = None
    if actual is not None:
        error = wer(actual, predicted_text)

    return predicted_text, error
