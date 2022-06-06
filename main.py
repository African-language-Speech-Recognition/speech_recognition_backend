from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def index():
    return {
        'status': 200,
        'message': 'Server Works'
    }


@app.post('/predict')
async def predict_audio():
    # TODO: load data from request
    # TODO: convert to spectrogram
    # TODO: predict
    # TODO: build response
    pass
