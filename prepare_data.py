import functools
import multiprocessing
import os
import pickle

import numpy as np
from tf import tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import constants
from dataset import Dataset
from midi import MIDI, Encoded


def dump_pickle(tps, paths, output_file):
    song_data = [x.data.T for x in get_pool().map(MIDI(tps).from_midi, paths)]
    pickle.dump(song_data, open(output_file, "wb"))
    print(f"Dumped '{output_file}' pickle")


def dump_all_midi_data(tps, composer, overwrite=False):
    """
    Dump all encoded midi objects to files
    :param tps: Ticks per second
    :param overwrite: If set to False this function will skip dumping to existing .pkl file
    :param composer: Composer of the midi files to pass to Dataset
    """
    d = Dataset(composer)
    train_pickle = os.path.join(constants.maestro_path, f"{composer}_{tps}tps_train.pkl")
    test_pickle = os.path.join(constants.maestro_path, f"{composer}_{tps}tps_test.pkl")
    validation_pickle = os.path.join(constants.maestro_path, f"{composer}_{tps}tps_validation.pkl")
    if overwrite or not os.path.isfile(train_pickle):
        dump_pickle(tps, d.train[1], train_pickle)
    if overwrite or not os.path.isfile(test_pickle):
        dump_pickle(tps, d.test[1], test_pickle)
    if overwrite or not os.path.isfile(validation_pickle):
        dump_pickle(tps, d.validation[1], validation_pickle)


def get_train_test_val_lists(tps, composer):
    dump_all_midi_data(tps, composer)

    train_pickle = os.path.join(constants.maestro_path, f"{composer}_{tps}tps_train.pkl")
    test_pickle = os.path.join(constants.maestro_path, f"{composer}_{tps}tps_test.pkl")
    validation_pickle = os.path.join(constants.maestro_path,
                                     f"{composer}_{tps}tps_validation.pkl")

    train = pickle.load(open(train_pickle, 'rb'))
    test = pickle.load(open(test_pickle, 'rb'))
    validation = pickle.load(open(validation_pickle, 'rb'))

    return train, test, validation


def get_notes_range(tps=None, composer=None, data=None):
    if data is None:
        if tps is None:
            print("tps parameter must be set if output is not set")
        train, test, validation = get_train_test_val_lists(tps, composer)
        data = train + test + validation
        data = np.concatenate(data, axis=0)
    distribution = data.sum(axis=0)
    # np.savetxt(f"output/{composer}_distribution.csv", distribution, delimiter=",")
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


def process(data, start, end, add_end_token=True):
    data = data[:, start:end]

    if add_end_token:
        extra = np.zeros((1, data.shape[1]), dtype=data.dtype)
        data = np.concatenate((data, extra), axis=0)
    extra_row = np.zeros((data.shape[0], 1), dtype=data.dtype)
    if add_end_token:
        extra_row[-1, 0] = 1
    data = np.concatenate((data, extra_row), axis=1)

    return data


def restore(data, start, end, remove_end_token=True):
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


def get_processed_data(tps, composer, dtype=np.float):
    train, test, validation = get_train_test_val_lists(tps, composer)
    start, end = get_notes_range(tps, composer)

    train = np.concatenate([process(data, start, end) for data in train], axis=0)
    test = np.concatenate([process(data, start, end) for data in test], axis=0)
    validation = np.concatenate([process(data, start, end) for data in validation], axis=0)

    train = train.astype(dtype)
    test = test.astype(dtype)
    validation = validation.astype(dtype)

    return (train, test, validation), (start, end)


def ts_generator(dataset, window_size: int) -> TimeseriesGenerator:
    return TimeseriesGenerator(dataset, dataset, window_size, shuffle=True)


def test_process_restore():
    m = MIDI()
    input_midi_name = 'unfin'
    encoded = m.from_midi(f'output/{input_midi_name}.midi')
    data = encoded.data.T
    start, end = get_notes_range(data=data)
    data = process(data, start, end)

    restored_data = restore(data, start, end)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"output/test_{input_midi_name}.midi")


def main():
    test_process_restore()


@functools.lru_cache
def get_pool():
    return multiprocessing.Pool()


if __name__ == '__main__':
    main()
