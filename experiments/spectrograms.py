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

GENERATE_SPEC_FOR = ["tcof"] # ["librispeech", "sitw", "common_voice", "tcof"]


def generate_filepath(dataset, subset, speaker_id, index):
    out = DATA_PATH + f'/{dataset}.spec/{subset}/{speaker_id}/{index}.npy'
    return out


if "librispeech" in GENERATE_SPEC_FOR:
    librispeech_subsets = ['dev-clean', 'train-clean-100', 'train-clean-360']
    dataset = 'LibriSpeech'
    for subset in librispeech_subsets:

        print(subset)
        waveforms = LibriSpeech(subset, seconds=None, down_sampling=1, stochastic=False, pad=False)
        spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=window_length, window_hop=window_hop)
        loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

        mkdir(DATA_PATH + f'/{dataset}.spec/')
        mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/')

        pbar = tqdm(total=len(loader))
        for i, (spec, y) in enumerate(loader):
            spec = spec.numpy()
            y = y.item()
            outpath = generate_filepath(dataset, subset, y, i)
            mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/{y}/')

            np.save(outpath, spec)
            pbar.update(1)
        pbar.close()

if "sitw" in GENERATE_SPEC_FOR:
    dataset = 'sitw'
    for split in ['dev', 'eval']:
        print(split)
        waveforms = SpeakersInTheWild(split, 'enroll-core', seconds=None, down_sampling=1, stochastic=False, pad=False)
        spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=window_length, window_hop=window_hop)
        loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

        mkdir(DATA_PATH + f'/{dataset}.spec/')
        mkdir(DATA_PATH + f'/{dataset}.spec/{split}/')

        pbar = tqdm(total=len(loader))
        for i, (spec, y) in enumerate(loader):
            spec = spec.numpy()
            y = y.item()
            outpath = generate_filepath(dataset, split, y, i)
            mkdir(DATA_PATH + f'/{dataset}.spec/{split}/{y}/')

            np.save(outpath, spec)
            pbar.update(1)
        pbar.close()

if "common_voice" in GENERATE_SPEC_FOR:
    common_voice_subsets = ['train', 'test']
    dataset = 'common_voice-fr'
    language = 'fr'
    for split in common_voice_subsets:
        print(split)
        waveforms = CommonVoice(language, split, seconds=None, down_sampling=1, stochastic=False, pad=False)
        spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=float(window_length/3), window_hop=float(window_hop/3))
        loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

        mkdir(DATA_PATH + f'/{dataset}.spec/')
        mkdir(DATA_PATH + f'/{dataset}.spec/{split}/')

        pbar = tqdm(total=len(loader))
        for i, (spec, y) in enumerate(loader):
            spec = spec.numpy()
            y = y.item()
            outpath = generate_filepath(dataset, split, y, i)
            mkdir(DATA_PATH + f'/{dataset}.spec/{split}/{y}/')

            np.save(outpath, spec)
            pbar.update(1)
        pbar.close()

if "tcof" in GENERATE_SPEC_FOR:
    tcof_subsets = ['train', 'test'] # ['dev']
    new_subset_name = "train"
    dataset = 'TCOF'
    waveforms = TCOF(tcof_subsets, seconds=None, down_sampling=1, stochastic=False, pad=False)
    spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=window_length, window_hop=window_hop)
    loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

    mkdir(DATA_PATH + f'/{dataset}.spec/')
    mkdir(DATA_PATH + f'/{dataset}.spec/{new_subset_name}/')

    pbar = tqdm(total=len(loader))
    for i, (spec, y) in enumerate(loader):
        spec = spec.numpy()
        y = y.item()
        outpath = generate_filepath(dataset, new_subset_name, y, i)
        mkdir(DATA_PATH + f'/{dataset}.spec/{new_subset_name}/{y}/')

        np.save(outpath, spec)
        pbar.update(1)

    pbar.close()
