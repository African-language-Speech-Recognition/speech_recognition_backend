from fastapi import FastAPI, UploadFile
from scripts.tokenizer import Tokenizer
from scripts.helper_functions import read_obj, extract_audio
from scripts.models import build_model, cnn_net, bi_directional_rnn, preprocess_model
from scripts.predict import predict
import pandas as pd

# load files
meta_data = pd.read_csv("./data/meta_data.csv")
sorted_metadata = meta_data.sort_values(by="duration")
labels = sorted_metadata['label'].to_list()
translation_obj = read_obj("./data/translation_dict.pkl")

# load translation
translations = []
for label in labels:
    translations.append(translation_obj[label])

# init tokenizer
tokenizer = Tokenizer(translations)
int_to_char, char_to_int = tokenizer.build_dict()
output_dim = len(char_to_int) + 2

# CNN
n_mels = 128
cnn_model, cnn_shape = cnn_net(n_mels)

# BI-DIRECTIONAL RNN
batch_size = 32
bi_rnn = bi_directional_rnn(1024, batch_size=batch_size, output_dim=output_dim)

# preprocessor
sample_rate = 8000
fft_size = 512
frame_step = 256
preprocess_model = preprocess_model(sample_rate, fft_size, frame_step, n_mels)

# build model
cnn_bi_rnn_model = build_model(output_dim, cnn_model, bi_rnn, preprocess_model)

# load saved model
cnn_bi_rnn_model.load_weights("./models/cnn-bi-rnn.h5")

app = FastAPI()


@app.get('/')
async def index():
    return {
        'status': 200,
        'message': 'Server Works'
    }


@app.post('/type')
async def get_type(file: UploadFile):
    return {
        'type': file.content_type,
    }


@app.post('/predict')
async def predict_audio(file: UploadFile):
    # load data from request
    audio_file = file.file

    # preprocess audio
    extracted_audio = extract_audio(audio_file)

    # predict
    predicted, error = predict(cnn_bi_rnn_model, extracted_audio[0],
                               tokenizer, int_to_char, actual=None)
    # build response
    return {
        "predicted": predicted
    }
