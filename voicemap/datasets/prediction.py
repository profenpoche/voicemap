import numpy as np
import pandas as pd
import soundfile as sf

from typing import Union, List

from torch.utils.data import Dataset
from voicemap.datasets.core import AudioDataset
from voicemap.datasets import SpectrogramDataset
from config import DATA_PATH

import subprocess
import time


class PredictionAudioDataset(AudioDataset):

    """Dataset object for prediction

    # Arguments
        filepaths: A list of the filepaths to predict
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        down_sampling:
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
    """


    base_sampling_rate = 16000

    def __init__(self,
                 filepaths: Union[str, List[str]],
                 seconds: Union[int, None],
                 down_sampling: int,
                 pad: bool = False,
                 stochastic: bool = True,
                 data_path: str = DATA_PATH):

        if isinstance(filepaths, str):
            filepaths = [filepaths]
    
        self.down_sampling = down_sampling
        self.seconds = seconds
        if seconds is not None:
            self.fragment_length = int(seconds * self.base_sampling_rate)
        self.pad = pad
        self.stochastic = stochastic
        self.df = pd.DataFrame(filepaths, columns=["filepath"])
        self.datasetid_to_filepath = self.df.to_dict()['filepath']

    def __len__(self):
        return len(self.df)

    def append(self, filepaths: Union[str, List[str]]):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        filepaths_res = []
        for filepath in filepaths:
            filepath = {'filepath': filepath}
            filepaths_res.append(filepath)
        self.df = self.df.append(filepaths_res, ignore_index=True)
        self.datasetid_to_filepath = self.df.to_dict()['filepath']

    @property
    def num_classes(self):
        return NotImplementedError

    def readAudioFile(self, filename):
        try:
            audioFileFormat = str(subprocess.check_output(['sox', '--i','-t', filename], stderr=subprocess.STDOUT).decode('utf-8').strip())
        except subprocess.CalledProcessError as error:
            print(error)
            raise NameError(f"The file '{filename}' was not found")
        if (audioFileFormat == "wav" or audioFileFormat == "flac"):
            instance, samplerate = sf.read(filename)
        else:
            raise Exception(f"The extension '{audioFileFormat}' is not supported for '{filename}'")
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
            except NameError as error:
                raise NameError(error)
            except Exception as error:
                print(error)
                subprocess.check_output(['lame', '-m', 'm', filename], stderr=subprocess.STDOUT)
                subprocess.check_output(['lame', '--decode', filename.replace('.wav', '.mp3')], stderr=subprocess.STDOUT)
                time.sleep(1)
                i+=1
            except:
                e = sys.exc_info()[0]
                print(e)
                i+=1
            
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

            label = -1 # It is prediction, speaker is unknown
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
