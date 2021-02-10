import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from lstm_settings import base_settings, get_model_id
from midi import MIDI, Encoded
from prepare_data import process, restore, get_notes_range
from tf import tf


def run_generation(settings):
    m = MIDI(settings.ticks_per_second)

    input_file = os.path.abspath('input/unfin.midi')
    input_name = os.path.splitext(os.path.basename(input_file))[0]

    encoded = m.from_midi(input_file)
    data = encoded.data.T
    note_range = get_notes_range(settings.ticks_per_second, settings.composer)
    input_data = process(data, *note_range, add_end_token=False)
    input_data[input_data > 0] = 1
    n_notes = input_data.shape[1]

    checkpoint_path = f"output/{get_model_id(settings, n_notes)}"
    classifier = tf.keras.models.load_model(checkpoint_path)
    print(f'Loaded {checkpoint_path} model')

    samples_data = generate(settings, classifier, input_data, n_notes,
                            f'{input_name}_{get_model_id(settings, n_notes)}')

    restored_data = restore(samples_data, *note_range, remove_end_token=False)

    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"output/{input_name}_{get_model_id(settings, n_notes)}.midi")

    # Combine original and restored
    combined = np.concatenate((data, restored_data))
    m.to_midi(Encoded(combined.T, *encoded[1:]),
              f"output/combined_{input_name}_{get_model_id(settings, n_notes)}.midi")


def generate(settings, classifier, input_data, n_notes, file_name):
    output_size = settings.window_size * 4
    output = np.empty((output_size, n_notes))
    output[:settings.window_size] = input_data[-settings.window_size:]
    start = time.perf_counter()
    for i in tqdm(range(settings.window_size, output_size)):
        input_window = np.expand_dims(output[:i], 0)
        sample = classifier(input_window)

        sample = np.array(sample)
        sample = (sample > settings.threshold_scale) * 100

        output[i] = sample

    print(f'Generated {output.shape[0]} ticks in {(time.perf_counter() - start):.2f}s')

    cv2.imwrite(f"output/raw_{file_name}.png", output.T * 255)

    # Remove input from result
    samples = output[settings.window_size:]

    # Machine Learning™
    # The model isn't very good at outputting zero, so we remove everything below this arbitrary threshold
    sample_max = np.max(samples)
    threshold = sample_max * settings.threshold_scale
    samples = (samples > threshold) * 100

    samples_data = samples.clip(0, 127).astype(np.int8)

    cv2.imwrite(f"output/samples_{file_name}.png", samples.T)
    return samples_data


if __name__ == '__main__':
    run_generation(base_settings)
