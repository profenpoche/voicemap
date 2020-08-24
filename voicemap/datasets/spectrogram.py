from .core import AudioDataset
from typing import Union, Callable, List

import librosa
import numpy as np
import pandas as pd
from config import DATA_PATH
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Wraps a waveform dataset to transform it into a spectrogram dataset.

    Args:
        dataset: Base audio dataset.
        normalisation: normalisation function name
        window_length: Length of STFT window in seconds.
        window_hop: Time in seconds between STFT windows.
        dataset_name: If given, load spect from a csv
        subsets_names: Names of the subsets to load from csv
        transform: Function to be applied on precompute spect before returning them
        window_type: Type of STFT window i.e. 'hamming'
    """
    def __init__(self,
                 dataset: AudioDataset,
                 normalisation: Union[str, None],
                 window_length: Union[float, None],
                 window_hop: Union[float, None],
                 dataset_name: Union[str, None] = None,
                 subsets_names: Union[str, List[str], None] = None,
                 transform: Union[Callable, None] = None,
                 window_type: Union[str, float, tuple, Callable] = 'hamming'):
        self.dataset = dataset
        self.df = self.dataset.df
        self.normalisation = normalisation
        self.window_length = window_length
        self.window_hop = window_hop
        self.dataset_name = dataset_name
        self.transform = transform
        self.window_type = window_type
        self.loader = np.load
        # If precompute spect then loads them from the csv
        if dataset_name is not None and subsets_names is not None:
            if isinstance(subsets_names, str):
                self.subsets_names = [subsets_names]
            else:
                self.subsets_names = subsets_names
            self.spec_df = pd.DataFrame()
            for subset_name in self.subsets_names:
                subset_spec_df = pd.read_csv(f"{DATA_PATH}/{dataset_name}.spec/{subset_name}.csv")
                self.spec_df = pd.concat([self.spec_df, subset_spec_df])
                print(f"Spectrograms loaded from '{DATA_PATH}/{dataset_name}.spec/{subset_name}.csv'")
            features = ['filepath', 'speaker_id']
            merged = pd.merge(self.df, self.spec_df, how="inner", on=features)
            features.extend(['spec_filepath', 'seconds'])
            self.spec_df = merged.loc[:, features]
            self.datasetid_to_spec_filepath = self.spec_df.to_dict()['spec_filepath']
            self.datasetid_to_speaker_id = self.spec_df.to_dict()['speaker_id']
            # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
            self.unique_speakers = sorted(self.spec_df['speaker_id'].unique())
            self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes)}

        print(f"{self.dataset} wrapped into SpectrogramDataset with (window_length, hop) = ({self.window_length},{self.window_hop})")

    def __len__(self):
        return len(self.dataset)

    def append(self, filepaths: Union[str, List[str]]):
        self.dataset.append(filepaths)
        self.df = self.dataset.df

    @property
    def num_classes(self):
        return self.dataset.num_classes

    def waveform_to_logmelspectrogam(self, waveform: np.ndarray):
        D = librosa.stft(waveform,
                         n_fft=int(self.dataset.base_sampling_rate * self.window_length),
                         hop_length=int(self.dataset.base_sampling_rate * self.window_hop),
                         window=self.window_type)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        if self.normalisation == 'global':
            spect = (spect - spect.mean()) / (spect.std() + 1e-6)
        elif self.normalisation == 'frequency':
            spect = (spect - spect.mean(axis=0, keepdims=True)) / (spect.std(axis=0, keepdims=True) + 1e-6)

        return spect

    def __getitem__(self, item):
        if self.dataset_name is not None:
            spectrogram = self.loader(self.datasetid_to_spec_filepath[item])
            if self.transform is not None:
                spectrogram = self.transform(spectrogram)
            label = self.datasetid_to_speaker_id[item]
            label = self.speaker_id_mapping[label]
        else:
            waveform, label = self.dataset[item]
            spectrogram = self.waveform_to_logmelspectrogam(waveform[0])
        return spectrogram, label
