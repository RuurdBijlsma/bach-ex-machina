from dataset import Dataset
import settings
from midi import MIDI, Encoded
import os
import pickle
import numpy as np


def dump_pickle(paths, output_file):
    midi_mapped = list(map(MIDI().from_midi, paths))
    pickle.dump(midi_mapped, open(output_file, "wb"))
    print(f"Dumped '{output_file}' pickle")


def dump_all_midi(composer, overwrite=False):
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


def get_encoded_data(composer):
    dump_all_midi(composer)

    train_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_train.pkl")
    test_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_test.pkl")
    validation_pickle = os.path.join(settings.MAESTRO_PATH, f"{composer}_validation.pkl")

    train = pickle.load(open(train_pickle, 'rb'))
    test = pickle.load(open(test_pickle, 'rb'))
    validation = pickle.load(open(validation_pickle, 'rb'))

    return train, test, validation


def get_notes_range(composer):
    train, test, validation = get_encoded_data(composer)
    data = train + test + validation
    distribution = np.concatenate([x.data for x in data], axis=1).sum(axis=1)
    # np.savetxt(f"data/{composer}_distribution.csv", distribution, delimiter=",")
    start = np.argmax(distribution > 0)
    end = distribution.size - np.argmax(distribution[::-1] > 0) - 1

    return start, end


def process(encoded, start, end, add_end_token=True):
    data = encoded.data[start:end, :]
    if add_end_token:
        data = np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)
        bottom_row = np.zeros((1, data.shape[1]))
        bottom_row[:, -1] = 1
        data = np.concatenate((data, bottom_row), axis=0)
    return Encoded(data, *encoded[1:])


def restore(encoded, start, end, remove_end_token=True):
    data = encoded.data
    if remove_end_token:
        data = data[0:data.shape[0] - 1, 0:data.shape[1] - 1]
    top = np.zeros((start, data.shape[1]), dtype=data.dtype)
    bottom = np.zeros((128 - end, data.shape[1]), dtype=data.dtype)
    data = np.concatenate((top, data, bottom), axis=0)
    return Encoded(data, *encoded[1:])


def to_input_output(data):
    pass  # merge all encoded datas here to mega 2d array of 128x50000000 and return that (x)
    # and the same shifted one to the right for output (y)


def get_processed_data(composer):
    train, test, validation = get_encoded_data(composer)
    start, end = get_notes_range(composer)

    train = [process(encoded, start, end) for encoded in train]
    test = [process(encoded, start, end) for encoded in test]
    validation = [process(encoded, start, end) for encoded in validation]

    return (train, test, validation), (start, end)


def main():
    composer = "bach"
    (train, test, validation), (start, end) = get_processed_data(composer)
    to_input_output(train.data)

    first_encoded = train[0]
    restored = restore(first_encoded, start, end)
    print(first_encoded, restored)


if __name__ == '__main__':
    main()
