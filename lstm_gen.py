import os
import time

import cv2
import numpy as np
import tensorflow as tf

from midi import MIDI, Encoded
from prepare_data import process, restore, get_notes_range


def main():
    m = MIDI(8)
    composer = 'bach'
    compress = 1
    window_size = 30

    input_file = os.path.abspath('data/unfin.midi')
    input_name = os.path.splitext(os.path.basename(input_file))[0]

    encoded = m.from_midi(input_file)
    data = encoded.data.T
    note_range = get_notes_range(composer)
    input_data = process(data, *note_range, compress, add_end_token=False)
    input_data[input_data > 0] = 1
    n_notes = input_data.shape[1]

    checkpoint_path = f"data/{composer}_checkpoint_n{n_notes}_c{compress}"

    classifier = tf.keras.models.load_model(checkpoint_path)

    output_size = input_data.shape[0] - window_size

    # Shape the input
    input_windows = np.array([
        input_data[i:i + window_size, :].astype(np.float32)
        for i
        in range(output_size)
    ])

    start = time.perf_counter()
    samples = classifier.predict(input_windows)
    print(f'Generated {samples.shape[0]} ticks in {(time.perf_counter() - start):.2f}s')

    samples *= 255
    output_name = f"{input_name}_{compress}_{n_notes}_{composer}"
    cv2.imwrite(f"data/lstm_samples_{output_name}.png", samples.T * 2)

    samples_data = samples.clip(0, 127).astype(np.int8)
    restored_data = restore(samples_data, *note_range, compress, remove_end_token=False)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"data/lstm_predicted_{output_name}.midi")


if __name__ == '__main__':
    main()
