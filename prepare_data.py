import functools
import multiprocessing
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import settings
from dataset import Dataset
from midi import MIDI, Encoded


def dump_pickle(paths, output_file):
    song_data = [x.data.T for x in get_pool().map(MIDI().from_midi, paths)]
    pickle.dump(song_data, open(output_file, "wb"))
    print(f"Dumped '{output_file}' pickle")


def dump_all_midi_data(composer, overwrite=False):
    """
    Dump all encoded midi objects to files
    :param overwrite: If set to False this function will skip dumping to existing .pkl file
    :param composer: Composer of the midi files to pass to Dataset
    """
    d = Dataset(composer)
    train_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_train.pkl")
    test_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_test.pkl")
    validation_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_validation.pkl")
    if overwrite or not os.path.isfile(train_pickle):
        dump_pickle(d.train[1], train_pickle)
    if overwrite or not os.path.isfile(test_pickle):
        dump_pickle(d.test[1], test_pickle)
    if overwrite or not os.path.isfile(validation_pickle):
        dump_pickle(d.validation[1], validation_pickle)


def get_train_test_val_lists(composer):
    dump_all_midi_data(composer)

    train_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_train.pkl")
    test_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_test.pkl")
    validation_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_validation.pkl")

    train = pickle.load(open(train_pickle, 'rb'))
    test = pickle.load(open(test_pickle, 'rb'))
    validation = pickle.load(open(validation_pickle, 'rb'))

    return train, test, validation


def get_notes_range(composer=None, data=None):
    if data is None:
        train, test, validation = get_train_test_val_lists(composer)
        data = train + test + validation
        data = np.concatenate(data, axis=0)
    distribution = data.sum(axis=0)
    # np.savetxt(f"data/{composer}_distribution.csv", distribution, delimiter=",")
    start = np.argmax(distribution > 0)
    end = distribution.size - np.argmax(distribution[::-1] > 0) - 1

    return start, end


def downscale_midi_array(data, factor):
    result = np.zeros((data.shape[0], data.shape[1] // factor), dtype=np.float64)

    for row_i in range(result.shape[1]):
        for j in range(factor):
            if j >= data.shape[1]:
                break
            result[:, row_i] += data[:, row_i * factor + j]

    result = result.clip(max=127).astype(np.int8)
    return result


def upscale_midi_array(data, factor):
    result = np.zeros((data.shape[0], data.shape[1] * factor), dtype=data.dtype)

    for row_i in range(data.shape[1]):
        result[:, row_i * factor] = data[:, row_i]

    return result


def process(data, start, end, compress=1, add_end_token=True):
    if compress != 1:
        data = downscale_midi_array(data, compress)
        start = start // compress
        end = end // compress
    data = data[:, start:end]

    if add_end_token:
        extra = np.zeros((1, data.shape[1]), dtype=data.dtype)
        data = np.concatenate((data, extra), axis=0)
    extra_row = np.zeros((data.shape[0], 1), dtype=data.dtype)
    if add_end_token:
        extra_row[-1, 0] = 1
    data = np.concatenate((data, extra_row), axis=1)

    return data


def restore(data, start, end, decompress=1, remove_end_token=True):
    if decompress != 1:
        data = upscale_midi_array(data, decompress)
        start = start // decompress * 2
        end = start + (data.shape[1] - (1 if remove_end_token else 0))
    if remove_end_token:
        data = data[0:data.shape[0] - 1, 0:data.shape[1] - 1]
    else:
        data = data[:, 0:data.shape[1] - 1]
    top = np.zeros((data.shape[0], start), dtype=data.dtype)
    bottom = np.zeros((data.shape[0], 128 - end), dtype=data.dtype)
    data = np.concatenate((top, data, bottom), axis=1)
    return data


def to_input_output(data):
    x = data[:-1, :]
    y = data[1:, :]

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1]))

    return x, y


def create_dataset(x, y, window_size):
    batch_size = 32
    return (tf.data.Dataset
            .from_tensor_slices((x, y))
            .window(window_size, 1, 1, False)
            .shuffle(1_000)
            .batch(batch_size)
            .prefetch(1)
            )


def get_processed_data(composer, compress=1, dtype=np.float):
    train, test, validation = get_train_test_val_lists(composer)
    start, end = get_notes_range(composer)

    train = np.concatenate([process(data, start, end, compress) for data in train], axis=0)
    test = np.concatenate([process(data, start, end, compress) for data in test], axis=0)
    validation = np.concatenate([process(data, start, end, compress) for data in validation], axis=0)

    train = train.astype(dtype)
    test = test.astype(dtype)
    validation = validation.astype(dtype)

    return (train, test, validation), (start, end)


def ts_generator(dataset, window_size: int) -> TimeseriesGenerator:
    return TimeseriesGenerator(dataset, dataset, window_size, shuffle=True)


def test_process_restore():
    m = MIDI()
    compress = 2
    input_midi_name = 'unfin'
    encoded = m.from_midi(f'data/{input_midi_name}.midi')
    data = encoded.data.T
    start, end = get_notes_range(data=data)
    data = process(data, start, end, compress)

    restored_data = restore(data, start, end, compress)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"data/compress_test_{input_midi_name}.midi")


def test_train_data():
    composer = "bach"
    compress = 2
    (train, test, validation), (start, end) = get_processed_data(composer, compress)
    (train_x, train_y) = to_input_output(train)
    (test_x, test_y) = to_input_output(test)
    (val_x, val_y) = to_input_output(validation)


def main():
    test_process_restore()


@functools.lru_cache
def get_pool():
    return multiprocessing.Pool()


if __name__ == '__main__':
    main()
