"""Preprocesses waveforms to create spectrogram datasets."""
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

from voicemap.datasets import (
    LibriSpeech,
    SpeakersInTheWild,
    CommonVoice,
    SpectrogramDataset,
    TCOF,
)
from voicemap.utils import mkdir, rmdir
from config import DATA_PATH
from voicemap.datasets.datasets import generate_waveforms_from_dataset_name, adjust_window_parameters


def make_dir(dataset_name, subset_name):
    mkdir(DATA_PATH + f"/{dataset_name}.spec/")
    mkdir(DATA_PATH + f"/{dataset_name}.spec/{subset_name}/")


def remove_dir(dataset_name):
    rmdir(DATA_PATH + f"/{dataset_name}.spec/")


def generate_filepath(dataset, subset, speaker_id, index):
    out = DATA_PATH + f"/{dataset}.spec/{subset}/{speaker_id}/{index}.npy"
    return out


def generate_loader_from_waveforms(
    waveforms, norm="global", window_length=0.02, window_hop=0.01
):
    """ Encapsulates the waveforms into a SpectrogramDataset, then creates a DataLoader
    """
    spectrograms = SpectrogramDataset(
        waveforms,
        normalisation=norm,
        window_length=window_length,
        window_hop=window_hop,
    )
    loader = DataLoader(spectrograms, batch_size=1, shuffle=False)
    return loader


def generate_spec_from_loader(loader, dataset, subset, verbose=True):
    """ Reads the spectrograms of the loader and saves them into the directory.
        Then saves the new filepaths into a csv
    """
    pbar = tqdm(total=len(loader))
    for i, (spec, y) in enumerate(loader):
        spec = spec.numpy()
        y = y.item()
        outpath = generate_filepath(dataset, subset, y, i)
        mkdir(DATA_PATH + f"/{dataset}.spec/{subset}/{y}/")
        loader.dataset.df.loc[i, "spec_filepath"] = outpath
        np.save(outpath, spec)
        pbar.update(1)
    loader.dataset.df.to_csv(DATA_PATH + f"/{dataset}.spec/{subset}.csv", index=False)
    if verbose:
        print("Spec index saved to " + DATA_PATH + f"/{dataset}.spec/{subset}.csv")
    pbar.close()


def compute_spectrograms(
    spec_to_generate,
    norm="global",
    window_length=0.02,
    window_hop=0.01,
    remove_old_dir=False,
    use_standardized=False,
    verbose=True,
):
    """ Generates and saves spectrograms for the datasets given in 'spec_to_generate'
        Also generates a csv indexing the new spec generated and their corresponding plain audio files
    """
    # Remove the spec directories if desired
    for dataset in spec_to_generate.keys():
        if remove_old_dir:
            remove_dir(dataset)
    # Generate and save spectrograms
    for dataset in spec_to_generate.keys():
        first_dataset_letter = dataset[0].upper()
        for subset in spec_to_generate[dataset]:
            try:
                # Create the directory
                make_dir(dataset, subset)
                # Load the waveforms
                waveforms = generate_waveforms_from_dataset_name(
                    first_dataset_letter,
                    subset,
                    seconds=None,
                    down_sampling=1,
                    use_standardized=use_standardized and subset == "train",
                    stochastic=False,
                    pad=False,
                )
                # Adjust spectrogram parameters according to the source waveform dataset
                window_l, window_h = adjust_window_parameters(window_length, window_hop, waveforms.base_sampling_rate)
                # Create the dataloader
                loader = generate_loader_from_waveforms(
                    waveforms, window_length=window_l, window_hop=window_h
                )
                # Generate the spectrograms and save it into an index csv file
                generate_spec_from_loader(loader, dataset, subset, verbose=verbose)
            except FileNotFoundError as error:
                print(error)


###################################
####           Main            ####
###################################

if __name__ == "__main__":
    # window_length = 0.02
    # window_hop = 0.01
    # norm = 'global'
    GENERATE_SPEC_FOR = {
        "LibriSpeech": ["train", "dev-clean"],
        "sitw": ["dev", "eval"],
        "common_voice-fr": ["train", "test"],
        "TCOF": ["train", "dev"],
    }
    if not use_standardized:
        GENERATE_SPEC_FOR["LibriSpeech"] = [
            "dev-clean",
            "train-clean-100",
            "train-clean-360",
        ]

    compute_spectrograms(GENERATE_SPEC_FOR, remove_old_dir=True, use_standardized=False)
