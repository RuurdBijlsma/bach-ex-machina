import os
from typing import Optional, Tuple, Iterable

import pandas as pd

import settings
from midi import MIDI

CSV_FILE = 'maestro-v3.0.0.csv'

TSplit = Tuple[Iterable[str], Iterable[str]]


class Dataset:
    def __init__(self,
                 composer: Optional[str] = None,
                 ):
        self.index = pd.read_csv(os.path.join(settings.MAESTRO_PATH, CSV_FILE))

        if composer is not None:
            self.index = self.index[self.index['canonical_composer'].str.contains(composer, case=False)]

    def _iter_split(self, split: str) -> TSplit:
        dataset = self.index[self.index['split'] == split]
        midi_paths = dataset['midi_filename'].map(lambda path: os.path.join(settings.MAESTRO_PATH, path))
        return dataset['canonical_title'], midi_paths

    @property
    def train(self) -> TSplit:
        return self._iter_split('train')

    @property
    def test(self) -> TSplit:
        return self._iter_split('test')

    @property
    def validation(self) -> TSplit:
        return self._iter_split('validation')


def main():
    ds = Dataset(composer='Bach')

    def count(gen):
        return sum(1 for _ in gen)

    print('train', count(ds.train[1]))
    print('test', count(ds.test[1]))
    print('val', count(ds.validation[1]))

    # Load files
    _, paths = ds.train

    for midi in map(MIDI().from_midi, paths):
        print(midi)


if __name__ == '__main__':
    main()
