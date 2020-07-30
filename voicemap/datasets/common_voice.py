from .core import AudioDataset
from typing import Union

import librosa
import numpy as np
import pandas as pd

from config import DATA_PATH

import soundfile as sf
import os


class CommonVoice(AudioDataset):
    base_sampling_rate = 48000

    def __init__(self,
                 language: str,
                 subset: str,
                 seconds: Union[int, None],
                 down_sampling: int,
                 sampling_rate: int = None,
                 stochastic: bool = True,
                 pad: bool = True,
                 data_path: str = DATA_PATH):
        self.language = language
        self.subset = subset
        self.seconds = seconds
        self.down_sampling = down_sampling
        self.data_path = data_path
        if sampling_rate is None:
            self.sampling_rate = self.base_sampling_rate
        else:
            if sampling_rate > self.base_sampling_rate:
                raise ValueError('Shouldn\'t have sampling rate higher than the sampling rate of the raw data.')
            self.sampling_rate = sampling_rate
            
        if seconds is not None:
            if int(seconds * self.base_sampling_rate) % down_sampling != 0:
                raise(ValueError, 'Down sampling must be an integer divisor of the fragment length.')

        if seconds is not None:
            self.fragment_length = int(seconds * self.base_sampling_rate)
        self.stochastic = stochastic
        self.pad = pad

        # if the {subset}_transformed.tsv is present we can load the data from it
        if (os.path.isfile(self.data_path + f'/CommonVoice/{self.language}/{self.subset}_transformed.tsv')):
            self.df = pd.read_csv(self.data_path + f'/CommonVoice/{self.language}/{self.subset}_transformed.tsv')
        else:
            self.df = pd.read_csv(self.data_path + f'/CommonVoice/{self.language}/{self.subset}.tsv', sep="\t")
            
            self.df['speaker_id'] = self.df['client_id']
            self.df['filepath'] = self.data_path + f'/CommonVoice/{self.language}/clips/' + self.df['path'] # + '.mp3'

            self.df['index'] = self.df.index.values

        # Trim too-small files
        if not self.pad and self.seconds is not None and 'seconds' in self.df.columns:
            self.df = self.df[self.df['seconds'] > self.seconds]

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.assign(id=self.df.index.values)

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']

        # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes)}

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        print("In CommonVoice "+ self.subset +" there are "+str(len(self.df['speaker_id'].unique()))+" speakers")
        return len(self.df['speaker_id'].unique())

    def __getitem__(self, index):
        instance, samplerate = librosa.core.load(self.datasetid_to_filepath[index], sr=self.sampling_rate)
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, max(len(instance) - self.fragment_length, 1))
        else:
            fragment_start_index = 0

        if self.seconds is not None:
            instance = instance[fragment_start_index:fragment_start_index + self.fragment_length]
            # print("fragment_length : "+str(fragment_start_index + self.fragment_length - fragment_start_index))
            # print("len instance before cut : "+str(len(instance)))
            # print("len instance : "+str(len(instance)))
        else:
            # Use whole sample
            pass
        # Check for required length and pad if necessary
        if hasattr(self, 'fragment_length'):
            if self.pad and len(instance) < self.fragment_length:
                less_timesteps = self.fragment_length - len(instance)
                if self.stochastic:
                    # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                    # before and the appropriate number of 0s after the instance
                    before_len = np.random.randint(0, less_timesteps)
                    after_len = less_timesteps - before_len

                    instance = np.pad(instance, (before_len, after_len), 'constant')
                else:
                    # Deterministic padding. Append 0s to reach self.fragment_length
                    instance = np.pad(instance, (0, less_timesteps), 'constant')

        label = self.datasetid_to_speaker_id[index]
        label = self.speaker_id_mapping[label]

        # Reindex to channels first format as supported by pytorch and downsample by desired amount
        instance = instance[np.newaxis, ::self.down_sampling]

        return instance, label
