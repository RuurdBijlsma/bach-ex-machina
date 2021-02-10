import os
import time

import cv2
import numpy as np

from lstm_settings import base_settings, get_model_id
from midi import MIDI, Encoded
from prepare_data import process, restore, get_notes_range

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)
])


def generate(settings):
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
    output_size = input_data.shape[0] - settings.window_size

    # Shape the input
    input_windows = np.array([
        input_data[i:i + settings.window_size, :].astype(np.float32)
        for i
        in range(output_size)
    ])

    start = time.perf_counter()
    samples = classifier.predict(input_windows)
    print(f'Generated {samples.shape[0]} ticks in {(time.perf_counter() - start):.2f}s')
    cv2.imwrite(f"output/raw_{input_name}_{get_model_id(settings, n_notes)}.png", samples.T * 255)

    # Machine Learning™
    # The model isn't very good at outputting zero, so we remove everything below this arbitrary threshold
    sample_max = np.max(samples)

    threshold = sample_max * settings.threshold_scale
    samples[samples < threshold] = 0

    scale = 100 / (sample_max or 1)
    print(f'Scaling values by {scale:.1f}')
    samples[samples > 0] = scale
    samples_data = samples.clip(0, 127).astype(np.int8)

    cv2.imwrite(f"output/samples_{input_name}_{get_model_id(settings, n_notes)}.png", samples.T)

    restored_data = restore(samples_data, *note_range, remove_end_token=False)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"output/{input_name}_{get_model_id(settings, n_notes)}.midi")


if __name__ == '__main__':
    generate(base_settings)
