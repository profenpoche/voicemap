"""Preprocesses waveforms to create spectogram datasets."""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, CommonVoice, SpectrogramDataset, TCOF
from voicemap.utils import mkdir
from config import DATA_PATH


window_length = 0.02
window_hop = 0.01
norm = 'global'

use_standardized = True
GENERATE_SPEC_FOR = ["librispeech","sitw","common_voice","tcof"] # ["librispeech", "sitw", "common_voice", "tcof"]
librispeech_subsets = ["train", "dev-clean"]
sitw_splits = ['dev', 'eval']
common_voice_subsets = ['train', 'test']
tcof_subsets = ['train', 'dev']


def generate_filepath(dataset, subset, speaker_id, index):
    out = DATA_PATH + f'/{dataset}.spec/{subset}/{speaker_id}/{index}.npy'
    return out

def generateSpecFromLoader(loader, dataset, subset):
    pbar = tqdm(total=len(loader))
    for i, (spec, y) in enumerate(loader):
        spec = spec.numpy()
        y = y.item()
        outpath = generate_filepath(dataset, subset, y, i)
        mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/{y}/')

        np.save(outpath, spec)
        pbar.update(1)
    pbar.close()

def generateLoaderFromWaveforms(waveforms):
    spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=window_length, window_hop=window_hop)
    loader = DataLoader(spectrograms, batch_size=1, shuffle=False)
    return loader

if "librispeech" in GENERATE_SPEC_FOR:
    if not use_standardized:
        librispeech_subsets = ['dev-clean', 'train-clean-100', 'train-clean-360']
    dataset = 'LibriSpeech'
    for subset in librispeech_subsets:
        print(subset)
        try:
            waveforms = LibriSpeech(subset, use_standardized=use_standardized and subset == "train", seconds=None, down_sampling=1, stochastic=False, pad=False)
            loader = generateLoaderFromWaveforms(waveforms)

            mkdir(DATA_PATH + f'/{dataset}.spec/')
            mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/')

            generateSpecFromLoader(loader, dataset, subset)
        except FileNotFoundError as error:
            print(error)

if "sitw" in GENERATE_SPEC_FOR:
    dataset = 'sitw'
    for split in sitw_splits:
        print(split)
        try:
            waveforms = SpeakersInTheWild(split, 'enroll-core', use_standardized=False, seconds=None, down_sampling=1, stochastic=False, pad=False)
            loader = generateLoaderFromWaveforms(waveforms)

            mkdir(DATA_PATH + f'/{dataset}.spec/')
            mkdir(DATA_PATH + f'/{dataset}.spec/{split}/')

            generateSpecFromLoader(loader, dataset, split)
        except FileNotFoundError as error:
            print(error)

if "common_voice" in GENERATE_SPEC_FOR:
    dataset = 'common_voice-fr'
    language = 'fr'
    for split in common_voice_subsets:
        print(split)
        try:
            waveforms = CommonVoice(language, split, use_standardized=use_standardized and split == "train", seconds=None, down_sampling=1, stochastic=False, pad=False)
            spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=float(window_length/3), window_hop=float(window_hop/3))
            loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

            mkdir(DATA_PATH + f'/{dataset}.spec/')
            mkdir(DATA_PATH + f'/{dataset}.spec/{split}/')

            generateSpecFromLoader(loader, dataset, split)
        except FileNotFoundError as error:
            print(error)

if "tcof" in GENERATE_SPEC_FOR:
    dataset = 'TCOF'
    for subset in tcof_subsets:
        try:
            waveforms = TCOF(tcof_subsets, use_standardized=use_standardized and subset == "train", seconds=None, down_sampling=1, stochastic=False, pad=False)
            loader = generateLoaderFromWaveforms(waveforms)

            mkdir(DATA_PATH + f'/{dataset}.spec/')
            mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/')

            generateSpecFromLoader(loader, dataset, subset)
        except FileNotFoundError as error:
            print(error)
