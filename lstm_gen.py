import os
import time
import settings
import cv2
import numpy as np
import tensorflow as tf

from midi import MIDI, Encoded
from prepare_data import process, restore, get_notes_range


def main():
    m = MIDI(settings.ticks_per_second)
    # composer = 'bach'
    composer = None
    window_size = 30

    input_file = os.path.abspath('data/unfin.midi')
    input_name = os.path.splitext(os.path.basename(input_file))[0]

    encoded = m.from_midi(input_file)
    data = encoded.data.T
    note_range = get_notes_range(composer)
    input_data = process(data, *note_range, add_end_token=False)
    input_data[input_data > 0] = 1
    n_notes = input_data.shape[1]

    checkpoint_path = f"data/{composer}_checkpoint_n{n_notes}_tps{settings.ticks_per_second}"

    classifier = tf.keras.models.load_model(checkpoint_path)
    print(f'Loaded {checkpoint_path} model')

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

    # Machine Learning™
    # The model isn't very good at outputting zero, so we remove everything below this arbitrary threshold
    sample_max = np.max(samples)
    threshold_scale = .8
    threshold = sample_max * threshold_scale
    samples[samples < threshold] = 0

    scale = 100 / (sample_max or 1)
    print(f'Scaling values by {scale:.1f}')
    samples *= scale

    output_name = f"{input_name}_{n_notes}_{composer}"
    cv2.imwrite(f"data/lstm_samples_{output_name}.png", samples.T * 2)

    samples_data = samples.clip(0, 127).astype(np.int8)
    restored_data = restore(samples_data, *note_range, remove_end_token=False)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"data/lstm_predicted_{output_name}.midi")


if __name__ == '__main__':
    main()
