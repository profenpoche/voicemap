import numpy as np
from typing import Union

from voicemap.datasets import (
    LibriSpeech,
    SpeakersInTheWild,
    ClassConcatDataset,
    SpectrogramDataset,
    DatasetFolder,
    CommonVoice,
    TCOF,
)
from voicemap.augment import SpecAugment
from config import PATH, DATA_PATH

# There is two similar dict but the 'datafolder' one is the necessary
# TODO : Gather the two dicts (remove letter_to_dataset_dict)
letter_to_dataset_dict = {
    "L": "LibriSpeech",
    "S": "sitw",
    "C": "common_voice-fr",
    "T": "TCOF",
}
letter_to_datafolder_dict = {
    "L": "LibriSpeech",
    "S": "sitw",
    "C": "CommonVoice",
    "T": "TCOF",
}

train_subsets_of_datasets = {
    "LibriSpeech": ["train-clean-100", "train-clean-360"],
    "sitw": ["dev"],
    "common_voice-fr": ["train"],
    "TCOF": ["train"],
}
train_standardized_subsets_of_datasets = {
    "LibriSpeech": ["train"],
    "sitw": ["dev"],
    "common_voice-fr": ["train"],
    "TCOF": ["train"],
}
val_subsets_of_datasets = {
    "LibriSpeech": ["dev-clean"],
    "sitw": ["eval"],
    "common_voice-fr": ["test"],
    "TCOF": ["dev"],
}


def random_crop(n, augmentation=lambda x: x, dim=1):
    def _random_crop(spect):
        start_index = np.random.randint(0, max(len(spect) - n, 1))

        # Zero pad
        if spect.shape[-1] < n:
            less_timesteps = n - spect.shape[-1]
            spect = np.pad(spect, ((0, 0), (0, 0), (0, less_timesteps)), "constant")

        if dim == 1:
            spect = spect[0, :, start_index : start_index + n]
        else:
            spect = spect[:, :, start_index : start_index + n]

        # Data augmentation
        spect = augmentation(spect)

        return spect

    return _random_crop


def adjust_window_parameters(
    window_length, window_hop, sampling_rate_not_adjusted, base_sampling_rate=16000
):
    """ Adjust window_length and hop if the sampling rate of the dataset is higher (and multiple) of the base sampling rate.
        For example common_voice has a rate of 48000 and librispeech one of 16000. The ratio is 3 so the window_length must be divided by 3.
    """
    sampling_rate_ratio = float(sampling_rate_not_adjusted / base_sampling_rate)
    if sampling_rate_ratio != 1:
        window_length = float(window_length / sampling_rate_ratio)
        window_hop = float(window_hop / sampling_rate_ratio)
    return window_length, window_hop


def generate_waveforms_from_dataset_name(
    dataset_name,
    subsets_names,
    seconds=None,
    down_sampling=1,
    use_standardized=False,
    stochastic=False,
    pad=False,
):
    """ Generates waveforms (dataset without spectrograms) for the different datasets listed in 'letter_to_dataset_dict'.
    """
    if isinstance(subsets_names, str):
        subsets_names = [subsets_names]
    sampling_rate_ratio_common_voice = int(
        CommonVoice.base_sampling_rate / LibriSpeech.base_sampling_rate
    )

    if dataset_name == "L" or dataset_name == letter_to_dataset_dict["L"]:
        waveforms = LibriSpeech(
            subsets_names,
            seconds,
            down_sampling,
            use_standardized=use_standardized,
            stochastic=stochastic,
            pad=pad,
        )
    elif dataset_name == "S" or dataset_name == letter_to_dataset_dict["S"]:
        waveforms = SpeakersInTheWild(
            subsets_names[0],
            "enroll-core",
            seconds,
            down_sampling,
            use_standardized=False,
            stochastic=stochastic,
            pad=pad,
        )
    elif dataset_name == "C" or dataset_name == letter_to_dataset_dict["C"]:
        waveforms = CommonVoice(
            "fr",
            subsets_names[0],
            seconds,
            int(down_sampling * sampling_rate_ratio_common_voice),
            use_standardized=use_standardized,
            stochastic=stochastic,
            pad=pad,
        )
    elif dataset_name == "T" or dataset_name == letter_to_dataset_dict["T"]:
        waveforms = TCOF(
            subsets_names,
            seconds,
            down_sampling,
            use_standardized=use_standardized,
            stochastic=stochastic,
            pad=pad,
        )
    else:
        raise NotImplementedError(
            f"Character {dataset_name} is not recognized as a dataset letter"
        )
    return waveforms


def wrap_dataset_into_spectrogram_dataset(
    letter,
    args,
    waveform_dataset,
    subsets_names,
    norm="global",
    transform=lambda x: x,
):
    """ Wraps the waveform dataset previously generated into a SpectrogramDataset.
        Handles precompute_spect and specificities of the datasets
    """
    # If not precompute_spect then calculate spec on the fly, else load from a spectrograms csv
    dataset_name = letter_to_dataset_dict[letter] if args.precompute_spect else None
    # Handle parameters specificity for some datasets
    window_length, window_hop = adjust_window_parameters(
        args.window_length, args.window_hop, waveform_dataset.base_sampling_rate
    )
    # Generate dataset
    dataset = SpectrogramDataset(
        dataset=waveform_dataset,
        dataset_name=dataset_name,
        subsets_names=subsets_names,
        normalisation=norm,
        window_length=window_length,
        window_hop=window_hop,
        transform=transform,
    )
    return dataset


def put_dataset_in_dict(datasets, dataset, dataset_letter):
    """ Append dataset to the dict of val datasets
    """
    datasets[letter_to_dataset_dict[dataset_letter]] = dataset
    return datasets


def gather_datasets(
    args,
    train_datasets_string="LSCT",
    val_datasets_string="LSCT",
    train_subsets_dict=train_subsets_of_datasets,
    val_subsets_dict=val_subsets_of_datasets,
):
    """ Returns a ClassConcatDataset gathering the datasets.
    Datasets can be SpectrogramDataset or not according to the parameters
         Creation of datasets - 3 possibilities :
    1. not spectrogram -> Load normally waveforms datasets from csv
    2. spectrogram not precompute -> Load normally waveforms datasets then calculate/encapsulate into SpectrogramDataset class
    3. precompute spectrograms -> Load from data/dataset.spec
    """
    if args.use_standardized:
        train_subsets_dict = train_standardized_subsets_of_datasets

    train_datasets = {}
    val_datasets = {}

    # 1. Load waveforms datasets from csv
    for letter in train_datasets_string:
        train_dataset = generate_waveforms_from_dataset_name(
            letter,
            train_subsets_dict[letter_to_dataset_dict[letter]],
            args.n_seconds,
            args.downsampling,
            args.use_standardized,
            stochastic=True,
            pad=False,
        )
        train_datasets = put_dataset_in_dict(train_datasets, train_dataset, letter)
    for letter in val_datasets_string:
        unseen_dataset = generate_waveforms_from_dataset_name(
            letter,
            val_subsets_dict[letter_to_dataset_dict[letter]],
            args.n_seconds,
            args.downsampling,
            use_standardized=False,
            stochastic=True,
            pad=False,
        )
        val_datasets = put_dataset_in_dict(val_datasets, unseen_dataset, letter)

    # 2. & 3. If spect is wanted, then wraps the previous waveforms datasets into SpectrogramDatasets
    if args.spectrogram:
        # Prepare crop function if loaded from plain precompute spectrograms
        if args.precompute_spect:
            # Def of augmentation and transform functions
            if args.n_f > 0 or args.n_t > 0:
                augmentation = SpecAugment(args.n_f, args.F, args.n_t, args.T)
            else:
                augmentation = lambda x: x
            transform = random_crop(
                int(args.n_seconds / args.window_hop), augmentation, args.dim
            )
        else:
            transform = lambda x: x

        # 2. & 3. precompute spect or on the fly spect according to the 'args.precompute_spect' param
        for letter in train_datasets_string:
            train_dataset = wrap_dataset_into_spectrogram_dataset(
                letter,
                args,
                train_datasets[letter_to_dataset_dict[letter]],
                train_subsets_dict[letter_to_dataset_dict[letter]],
                transform=transform,
            )
            train_datasets = put_dataset_in_dict(train_datasets, train_dataset, letter)
        for letter in val_datasets_string:
            unseen_dataset = wrap_dataset_into_spectrogram_dataset(
                letter,
                args,
                val_datasets[letter_to_dataset_dict[letter]],
                val_subsets_dict[letter_to_dataset_dict[letter]],
                transform=transform,
            )
            val_datasets = put_dataset_in_dict(val_datasets, unseen_dataset, letter)

    train_datasets = ClassConcatDataset(list(train_datasets.values()))

    return train_datasets, val_datasets
