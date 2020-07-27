from .core import AudioDataset
from typing import Union, List

import numpy as np
import pandas as pd
import soundfile as sf
import subprocess

from config import DATA_PATH

import os
from os import path

import sys
import time


class TCOF(AudioDataset):
    """Dataset class representing the TCOF dataset (http://coalea.univ-lorraine.fr/content/traitement-de-corpus-oraux-en-francais-tcof/traitement-de-corpus-oraux-en-francais-tcof).

    # Arguments
        subsets: What subset of the TCOF dataset to use.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        down_sampling:
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
    """

    base_sampling_rate = 16000

    def __init__(self,
                 subsets: Union[str, List[str]],
                 seconds: Union[int, None],
                 down_sampling: int,
                 stochastic: bool = True,
                 pad: bool = False,
                 data_path: str = DATA_PATH):
        self.subsets = subsets
        self.seconds = seconds
        if seconds is not None:
            self.fragment_length = int(seconds * self.base_sampling_rate)
        self.down_sampling = down_sampling
        self.stochastic = stochastic
        self.pad = pad
        self.data_path = data_path

        if isinstance(subsets, str):
            subsets = [subsets]
    
        self.df = pd.DataFrame()
        for subset in subsets:
            # Get dataset info
            cached_df = pd.read_csv(self.data_path + f'/TCOF/TCOF_{subset}.csv',
                              delimiter=',', names=['speaker_id', 'filepath', 'transcript', 'wav_filesize'],header=1)
            self.df = pd.concat([cached_df,self.df])

        # Trim too-small files
        if not self.pad and self.seconds is not None:
            # For an audio with 16000 sample rate and 16-bit encoding (mono) there is 32000kB per second (add 2000 to be sure)
            self.df = self.df[self.df['wav_filesize'] > 34000 * self.seconds]
        
        # Remove left part of speaker_id (ex : <[speaker_folder_spk85_id_]>42)
        self.df['speaker_id'] = self.df['speaker_id'].transform(lambda x: int(x.split('_id_')[1]))

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
        print("In TCOF ", self.subsets, " there are "+str(len(self.df['speaker_id'].unique()))+" speakers")
        return len(self.df['speaker_id'].unique())

    def readAudioFile(self, filename):
        audioFileFormat = str(subprocess.check_output(['sox', '--i','-t', filename], stderr=subprocess.STDOUT).decode('utf-8').strip())
        if (audioFileFormat == "wav"):
            instance, samplerate = sf.read(filename)
        else:
            raise Exception(f"The extension '{audioFileFormat}' is not supported for TCOF dataset")
        duration = float(subprocess.check_output(['sox', '--i','-D', filename], stderr=subprocess.STDOUT).decode('utf-8').strip())
        if (self.seconds is not None and duration < self.seconds):
            raise Exception(f"The file '{filename}' has a duration less than '{self.seconds}' seconds")
        return instance, samplerate

    def __getitem__(self, index):
        filename = self.datasetid_to_filepath[index]
        # Some records are actually .mp3, so we have to handle it
        i=0
        errorAppeared = True
        while (errorAppeared and i < 5):
            try:
                instance, samplerate = self.readAudioFile(filename)
                errorAppeared = False
            except TypeError as error:
                print(error)
                subprocess.check_output(['lame', '--decode', filename], stderr=subprocess.STDOUT)
                i+=1
            except Exception as error:
                print(error)
                subprocess.check_output(['lame', '-m', 'm', filename], stderr=subprocess.STDOUT)
                subprocess.check_output(['lame', '--decode', filename.replace('.wav', '.mp3')], stderr=subprocess.STDOUT)
                time.sleep(1)
                i+=1
            except:
                e = sys.exc_info()[0]
                print(e)
                subprocess.check_output(['lame', '--decode', filename], stderr=subprocess.STDOUT)
                i+=1
        # print("=========================== i =",i,"for index",index)
        # print("len(instance)", len(instance))
            
        # Choose a random sample of the file
        try:
            if self.stochastic:
                fragment_start_index = np.random.randint(0, max(len(instance) - self.fragment_length, 1))
            else:
                fragment_start_index = 0

            if self.seconds is not None:
                instance = instance[fragment_start_index:fragment_start_index + self.fragment_length]
            else:
                # Use whole sample
                pass

            if hasattr(self, 'fragment_length'):
                # Check for required length and pad if necessary
                if self.pad and len(instance) < self.fragment_length:
                    less_timesteps = self.fragment_length - len(instance)
                    if self.stochastic:
                        # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                        # before and the appropriate number of 0s after the instance
                        less_timesteps = self.fragment_length - len(instance)

                        before_len = np.random.randint(0, less_timesteps)
                        after_len = less_timesteps - before_len

                        instance = np.pad(instance, (before_len, after_len), 'constant')
                    else:
                        # Deterministic padding. Append 0s to reach self.fragment_length
                        instance = np.pad(instance, (0, less_timesteps), 'constant')

            label = self.datasetid_to_speaker_id[index]
            label = self.speaker_id_mapping[label]

            # # Extract the first channel (necessary if the file is double-channel, not usual)
            if len(instance.shape) > 1:
                instance = np.transpose(instance)
                instance = instance[0]
                instance = np.transpose(instance)

            # Reindex to channels first format as supported by pytorch and downsample by desired amount
            instance = instance[np.newaxis, ::self.down_sampling]

            return instance, label
        except:
            print("Impossible to getitem", index)
            return [[0]*12000], 0 
