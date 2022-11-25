import os
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class RWCPSpikesDataset(Dataset):
    def __init__(self,
                 train_mode: bool = True,
                 train_size: float = 6 / 7,
                 slice_in_samples: int = None,
                 overlap: float = 0,
                 seed: int = 42):
        self.train_mode = train_mode
        self.base_dir = Path(os.path.dirname(__file__))
        if overlap >= 1 or overlap < 0:
            raise ValueError('Overlap argument can be a value in range of [0, 1)')
        metadata = pd.read_csv(self.base_dir / 'RWCP_spikes' / 'meta_data.csv')
        if slice_in_samples is None:
            slice_in_samples = metadata['size']
        self.encodes, self.labels = pd.factorize(metadata['label'])
        metadata['encode_label'] = self.encodes
        self.slice_in_samples = slice_in_samples
        self.classes = metadata['label'].unique()
        self._shuffle_files(metadata, seed)
        self.train_meta_data, self.test_meta_data = train_test_split(metadata, train_size=train_size)

        self.train_meta_data = self.prepare_data(self.train_meta_data, overlap)
        self.test_meta_data = self.prepare_data(self.test_meta_data, overlap)

        self.train_meta_data.reset_index(drop=True, inplace=True)
        self.test_meta_data.reset_index(drop=True, inplace=True)

        self._shuffle_files(self.train_meta_data, seed)

    def __len__(self):
        if self.train_mode:
            return len(self.train_meta_data)
        return len(self.test_meta_data)

    def __getitem__(self, idx):
        if self.train_mode:
            file = self.train_meta_data.iloc[idx]
        else:
            file = self.test_meta_data.iloc[idx]

        output_spikes = np.load(self.base_dir /
                                'RWCP_spikes' /
                                file['label'] / file['file_name'])
        output_spikes = np.array(list(dict(output_spikes).values()))
        output_spikes = output_spikes[:, file['start_index']:file['end_index']]
        return {
            'encoded_spikes': output_spikes,
            'label': file['encode_label']
        }

    @staticmethod
    def _shuffle_files(df, seed):
        return df.sample(frac=1, random_state=seed)

    def prepare_data(self, df, overlap):
        df['samples'] = df['size'] // (self.slice_in_samples * (1 - overlap))
        df = df.loc[
            df.index.repeat(df['samples'])
        ]
        df['start_index'] = (df
                             .groupby(['label', 'file_name'])
                             .cumcount() * (self.slice_in_samples * (1 - overlap))).astype(np.int64)
        df['end_index'] = df['start_index'] + self.slice_in_samples
        df['original_size'] = df['size']
        return df.drop(['size', 'samples'], axis=1)

    def label_encode_to_str(self, code):
        return self.labels[code]
