import os
from typing import Optional, Tuple, Iterable

import pandas as pd

import settings

CSV_FILE = 'maestro-v3.0.0.csv'


class Dataset:
    def __init__(self,
                 composer: Optional[str] = None,
                 ):
        self.index = pd.read_csv(os.path.join(settings.MAESTRO_PATH, CSV_FILE))

        if composer is not None:
            self.index = self.index[self.index['canonical_composer'].str.contains(composer)]

    def _iter_split(self, split: str) -> Iterable[Tuple[str, str]]:
        dataset = self.index[self.index['split'] == split]
        midi_paths = dataset['midi_filename'].map(lambda path: os.path.join(settings.MAESTRO_PATH, path))
        return zip(dataset['canonical_title'], midi_paths)

    @property
    def train(self) -> Iterable[Tuple[str, str]]:
        return self._iter_split('train')

    @property
    def test(self) -> Iterable[Tuple[str, str]]:
        return self._iter_split('test')

    @property
    def validation(self) -> Iterable[Tuple[str, str]]:
        return self._iter_split('validation')


if __name__ == '__main__':
    ds = Dataset(composer='Bach')


    def count(gen):
        return sum(1 for _ in gen)


    print('train', count(ds.train))
    print('test', count(ds.test))
    print('val', count(ds.validation))

    for name, path in ds.train:
        print(f'{name}:\t{path}')
