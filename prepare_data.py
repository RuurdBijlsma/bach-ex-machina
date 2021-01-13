from dataset import Dataset
import settings
from midi import MIDI, Encoded
import os
import pickle
import numpy as np
import functools
import multiprocessing


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


def get_notes_range(composer):
    train, test, validation = get_train_test_val_lists(composer)
    distribution = np.concatenate(train + test + validation, axis=0).sum(axis=0)
    # np.savetxt(f"data/{composer}_distribution.csv", distribution, delimiter=",")
    start = np.argmax(distribution > 0)
    end = distribution.size - np.argmax(distribution[::-1] > 0) - 1

    return start, end


def process(data, start, end, add_end_token=True):
    data = data[:, start:end]
    if add_end_token:
        extra = np.zeros((1, data.shape[1]))
        data = np.concatenate((data, extra), axis=0)
        extra_row = np.zeros((data.shape[0], 1))
        extra_row[-1, 0] = 1
        data = np.concatenate((data, extra_row), axis=1)
    return data


def restore(data, start, end, remove_end_token=True):
    if remove_end_token:
        data = data[0:data.shape[0] - 1, 0:data.shape[1] - 1]
    top = np.zeros((start, data.shape[1]), dtype=data.dtype)
    bottom = np.zeros((128 - end, data.shape[1]), dtype=data.dtype)
    data = np.concatenate((top, data, bottom), axis=1)
    return data


def to_input_output(data):
    x = data[:-1, :]
    y = data[1:, :]

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1]))

    return x, y


def get_processed_data(composer):
    train, test, validation = get_train_test_val_lists(composer)
    start, end = get_notes_range(composer)

    train = np.concatenate([process(data, start, end) for data in train], axis=0)
    test = np.concatenate([process(data, start, end) for data in test], axis=0)
    validation = np.concatenate([process(data, start, end) for data in validation], axis=0)

    return (train, test, validation), (start, end)


def main():
    composer = "bach"
    (train, test, validation), (start, end) = get_processed_data(composer)
    (train_x, train_y) = to_input_output(train)
    (test_x, test_y) = to_input_output(test)
    (val_x, val_y) = to_input_output(validation)


@functools.lru_cache
def get_pool():
    return multiprocessing.Pool()


if __name__ == '__main__':
    main()
