import os
import time

import cv2
import matplotlib.pyplot as plt
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
    output = np.zeros((output_size, n_notes))
    output[:settings.window_size] = input_data[-settings.window_size:]

    start = time.perf_counter()

    notes_per_tick = np.sum(input_data, axis=1)
    max_notes = np.max(notes_per_tick)
    mean_notes = np.mean(notes_per_tick)

    note_history = []

    for i in tqdm(range(settings.window_size, output_size)):
        start_window = i - settings.window_size
        input_window = np.expand_dims(output[start_window:i], 0)

        sample = classifier(input_window)
        sample = np.array(sample)
        sample_max = np.max(sample)

        candidates = sample[sample > sample_max - (2 * np.std(sample))].shape[0]
        last_step_notes = np.sum(input_window[:, -1])

        a = np.mean([candidates] * 2 + [last_step_notes, mean_notes])
        active_notes = int(min(np.round(a), max_notes))

        # plt.plot(np.squeeze(sample))
        # plt.title(f'{candidates} notes')
        # plt.show()

        # Remove most unlikely notes TODO does this help more than it hurts?
        sample[sample < sample_max * .005] = 0

        # Encourage keeping the same note
        sample += sample * input_window[:, -1] * 2

        # Convert the output to a probability vector, and use it to pick `active_notes` notes.
        note_probability = np.squeeze(sample)
        note_probability = note_probability / np.sum(note_probability)
        result = np.random.choice(n_notes, active_notes, replace=False, p=note_probability)

        # probably a better way to do this
        for v in result:
            output[i, v] = 1

        note_history.append(active_notes)

    print(f'Generated {output.shape[0]} ticks in {(time.perf_counter() - start):.2f}s')

    rgb = cv2.cvtColor(output.astype(np.float32).T * 255, cv2.COLOR_GRAY2BGR)
    cv2.line(rgb, (settings.window_size, 0), (settings.window_size, 128), color=(0, 0, 255))
    cv2.imwrite(f"output/raw_{file_name}.png", rgb)

    plt.plot(note_history)
    plt.title('Active notes')
    plt.show()

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
