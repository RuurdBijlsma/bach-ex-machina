import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from lstm_model import get_model
from prepare_data import process, restore, get_processed_data
from midi import MIDI, Encoded
import cv2
import numpy as np


def main():
    m = MIDI(8)
    composer = 'bach'
    compress = 1
    window_size = 30

    input_file = os.path.abspath('data/unfin.midi')
    input_name = os.path.splitext(os.path.basename(input_file))[0]

    encoded = m.from_midi(input_file)
    data = encoded.data.T
    _, (start, end) = get_processed_data(composer, compress)
    input_data = process(data, start, end, compress, add_end_token=False)
    input_data[input_data > 0] = 1
    n_notes = input_data.shape[1]

    print(input_data.shape)

    checkpoint_path = f"data/{composer}_checkpoint_n{n_notes}_c{compress}.ckpt"

    classifier = get_model(n_notes, window_size)
    classifier.load_weights(checkpoint_path)

    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    # tf.debugging.set_log_device_placement(True)

    samples_threshold = 0.5
    # inp = tf.keras.preprocessing.sequence.TimeseriesGenerator(input_data.T, input_data.T, window_size)
    output_size = input_data.shape[0] - window_size
    samples = np.empty((output_size, n_notes))
    for i in range(output_size):
        print(f'{i}/{output_size}')
        chunk = input_data[i:i + window_size, :]
        samples[i] = classifier(np.expand_dims(chunk, 0))

    # samples = None
    # samples[samples < samples_threshold] = 0
    # samples[samples > 0] = 100
    samples *= 255
    output_name = f"{input_name}_{compress}_{n_notes}_{composer}"
    cv2.imwrite(f"data/lstm_samples_{output_name}.png", samples.T * 2)

    print(samples)
    samples_data = samples.clip(0, 127).astype(np.int8)
    restored_data = restore(samples_data, start, end, compress, remove_end_token=False)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"data/lstm_predicted_{output_name}.midi")


if __name__ == '__main__':
    main()
