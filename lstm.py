import numpy as np
from midi import from_midi, to_midi, Encoded
import pickle
import cv2
from os import path

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
import dataset

def fit_model(data, model):
    model.fit(data)
    return model


def create_model(input_data, n_components, input_name):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_data.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_components))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    lstm = model.fit(input_data, epochs=200, batch_size=32)

    with open(f"data/lstm_{input_name}_{n_components}.pkl", "wb") as file:
        pickle.dump(lstm, file)
        print("Exported model to file!")
    return lstm


def load_model(n_components, input_name):
    with open(f"data/lstm_{input_name}_{n_components}.pkl", "rb") as file:
        print("Imported model from file!")
        return pickle.load(file)


def main():

    n_components = 35
    samples_threshold = 0.97

    ds = dataset.Dataset(composer='Bach')
    _, paths = ds.train
    input_data = map(from_midi, paths)
    input_data =np.asarray(input_data)
    model = create_model(input_data, n_components, input_name)
    # model = load_model(n_components, input_name)
    pred_input_file = path.abspath('data/unfin.midi')
    pred_input_name = path.splitext(path.basename(pred_input_file))[0]

    encoded_pred_in = from_midi(pred_input_file)

    prediction_input = encoded_pred_in.data.T

    prediction = model.predict(prediction_input, verbose=0)

    prediction[prediction < samples_threshold] = 0
    samples = prediction * 127
    cv2.imwrite(f"data/lstm_samples_{input_name}_{n_components}.png", samples.T * 2)

    print(prediction)
    pred_data = np.rint(prediction.T).astype(int).clip(0, 127)
    pred_encoded = Encoded(pred_data, encoded_pred_in.key_signature, encoded_pred_in.time_signature, encoded_pred_in.bpm)
    to_midi(pred_encoded, f"data/predicted_{pred_input_name}_{n_components}.midi")
    print('done')

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     main()
