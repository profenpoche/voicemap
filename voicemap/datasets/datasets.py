import numpy as np

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, ClassConcatDataset, SpectrogramDataset, DatasetFolder, CommonVoice, TCOF
from voicemap.augment import SpecAugment
from config import PATH, DATA_PATH

letter_to_dataset_dict = {
    'L': 'librispeech',
    'S': 'sitw',
    'C': 'common_voice',
    'T': 'tcof'
}


def append_datasets_train(train_datasets, train_dataset, dataset_letter):
    """ Append train_dataset & val_dataset to the list of train datasets & dict of val datasets
    """
    train_datasets.append(train_dataset)
    return train_datasets


def append_datasets_val(val_datasets, val_dataset, dataset_letter):
    """ Append train_dataset & val_dataset to the list of train datasets & dict of val datasets
    """
    val_datasets[letter_to_dataset_dict[dataset_letter]] = val_dataset
    return val_datasets

def gather_datasets(args, train_datasets_string="LSCT", val_datasets_string="LSCT"):
    """ Return a ClassConcatDataset gathering the datasets.
    Datasets can be SpectrogramDataset or not according to the parameters
    """
    train_datasets = []
    val_datasets = {}

    if args.use_standardized:
        librispeech_subsets = ['train']
    else:
        librispeech_subsets = ['train-clean-100', 'train-clean-360']
    unseen_subset = 'dev-clean'
    sitw_unseen = 'eval'
    tcof_subsets = ['train', 'test']

    sampling_rate_ratio_common_voice = int(CommonVoice.base_sampling_rate / LibriSpeech.base_sampling_rate)

    # Creation of datasets - 3 possibilities :
    # 1. precompute spectrograms -> Load from data/dataset.spec
    # 2. spectrogram not precompute -> Load normally then calculate/encapsulate into SpectrogramDataset class
    # 3. not sprectrogram -> Load normally from csv
    if args.spectrogram:
        if args.precompute_spect:
            # Def of augmentation and random crop
            if args.n_f > 0 or args.n_t > 0:
                augmentation = SpecAugment(args.n_f, args.F, args.n_t, args.T)
            else:
                augmentation = lambda x: x

            def random_crop(n):
                def _random_crop(spect):
                    start_index = np.random.randint(0, max(len(spect)-n, 1))

                    # Zero pad
                    if spect.shape[-1] < n:
                        less_timesteps = n - spect.shape[-1]
                        spect = np.pad(spect, ((0, 0), (0, 0), (0, less_timesteps)), 'constant')

                    if args.dim == 1:
                        spect = spect[0, :, start_index:start_index+n]
                    else:
                        spect = spect[:, :, start_index:start_index + n]

                    # Data augmentation
                    spect = augmentation(spect)

                    return spect

                return _random_crop
            transform = random_crop(int(args.n_seconds / args.window_hop))

            # 1. Precompute spectrograms
            for letter in train_datasets_string:
                if letter == 'L':
                    train_dataset = ClassConcatDataset([
                        DatasetFolder(
                            DATA_PATH + f'/LibriSpeech.spec/{subset}/', extensions='.npy', loader=np.load, transform=transform)
                        for subset in librispeech_subsets
                    ])
                elif letter == 'S':
                    train_dataset = DatasetFolder(DATA_PATH + '/sitw.spec/dev/', extensions='.npy', loader=np.load, transform=transform)
                elif letter == 'C':
                    train_dataset = DatasetFolder(DATA_PATH + '/common_voice-fr.spec/train/', extensions='.npy', loader=np.load, transform=transform)
                elif letter == 'T':
                    train_dataset = DatasetFolder(DATA_PATH + '/TCOF.spec/train/', extensions='.npy', loader=np.load, transform=transform)
                else:
                    raise NotImplementedError(f"Character {letter} is not recognized as a dataset letter")
                train_datasets = append_datasets_train(train_datasets, train_dataset, letter)

            for letter in val_datasets_string:
                if letter == 'L':
                    unseen_dataset = DatasetFolder(DATA_PATH + f'/LibriSpeech.spec/{unseen_subset}/', extensions='.npy',
                                                loader=np.load, transform=transform)
                elif letter == 'S':
                    unseen_dataset = DatasetFolder(DATA_PATH + f'/sitw.spec/{sitw_unseen}/', extensions='.npy', loader=np.load, transform=transform)
                elif letter == 'C':
                    unseen_dataset = DatasetFolder(DATA_PATH + '/common_voice-fr.spec/test/', extensions='.npy', loader=np.load, transform=transform)
                elif letter == 'T':
                    unseen_dataset = DatasetFolder(DATA_PATH + f'/TCOF.spec/dev/', extensions='.npy',
                                                loader=np.load, transform=transform)
                else:
                    raise NotImplementedError(f"Character {letter} is not recognized as a dataset letter")
                val_datasets = append_datasets_val(val_datasets, unseen_dataset, letter)
        else:
            # 2. Calculate spectrograms on the fly
            for letter in train_datasets_string:
                if letter == 'L':
                    train_dataset = SpectrogramDataset(
                        LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, use_standardized=args.use_standardized, stochastic=True, pad=False),
                        normalisation='global',
                        window_length=args.window_length,
                        window_hop=args.window_hop
                    )
                elif letter == 'S':
                    train_dataset = SpectrogramDataset(
                        SpeakersInTheWild('dev', 'enroll-core', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False),
                        normalisation='global',
                        window_length=args.window_length,
                        window_hop=args.window_hop
                    )
                elif letter == 'C':
                    train_dataset = SpectrogramDataset(
                        CommonVoice('fr', 'train', args.n_seconds, args.downsampling, use_standardized=args.use_standardized, stochastic=True, pad=True),
                        normalisation='global',
                        window_length=float(args.window_length / sampling_rate_ratio_common_voice),
                        window_hop=args.window_hop
                    )
                elif letter == 'T':
                    train_dataset = SpectrogramDataset(
                        TCOF(tcof_subsets, args.n_seconds, args.downsampling, use_standardized=args.use_standardized, stochastic=True, pad=False),
                        normalisation='global',
                        window_length=args.window_length,
                        window_hop=args.window_hop
                    )
                else:
                    raise NotImplementedError(f"Character {letter} is not recognized as a dataset letter")
                train_datasets = append_datasets_train(train_datasets, train_dataset, letter)
            for letter in val_datasets_string:
                if letter == 'L':
                    unseen_dataset = SpectrogramDataset(
                        LibriSpeech(unseen_subset, args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False),
                        normalisation='global',
                        window_length=args.window_length,
                        window_hop=args.window_hop
                    )
                elif letter == 'S':
                    unseen_dataset = SpectrogramDataset(
                        SpeakersInTheWild(sitw_unseen, 'enroll-core', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False),
                        normalisation='global',
                        window_length=args.window_length,
                        window_hop=args.window_hop
                    )
                elif letter == 'C':
                    unseen_dataset = SpectrogramDataset(
                        CommonVoice('fr', 'test', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=True),
                        normalisation='global',
                        window_length=float(args.window_length / sampling_rate_ratio_common_voice),
                        window_hop=args.window_hop
                    )
                elif letter == 'T':
                    unseen_dataset = SpectrogramDataset(
                        TCOF('dev', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False),
                        normalisation='global',
                        window_length=args.window_length,
                        window_hop=args.window_hop
                    )
                else:
                    raise NotImplementedError(f"Character {letter} is not recognized as a dataset letter")
                val_datasets = append_datasets_val(val_datasets, unseen_dataset, letter)
    else:
        # 3. Load from csv
        for letter in train_datasets_string:
            if letter == 'L':
                train_dataset = LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, use_standardized=args.use_standardized, stochastic=True, pad=False)
            elif letter == 'S':
                train_dataset = SpeakersInTheWild('dev', 'enroll-core', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False)
            elif letter == 'C':
                train_dataset = CommonVoice('fr', 'train', args.n_seconds, int(args.downsampling * sampling_rate_ratio_common_voice), use_standardized=args.use_standardized, stochastic=True, pad=True)
            elif letter == 'T':
                train_dataset = TCOF(tcof_subsets, args.n_seconds, args.downsampling, use_standardized=args.use_standardized, stochastic=True, pad=False)
            else:
                raise NotImplementedError(f"Character {letter} is not recognized as a dataset letter")
            train_datasets = append_datasets_train(train_datasets, train_dataset, letter)
        for letter in val_datasets_string:
            if letter == 'L':
                unseen_dataset = LibriSpeech(unseen_subset, args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False)
            elif letter == 'S':
                unseen_dataset = SpeakersInTheWild(sitw_unseen, 'enroll-core', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False)
            elif letter == 'C':
                unseen_dataset = CommonVoice('fr', 'test', args.n_seconds, int(args.downsampling * sampling_rate_ratio_common_voice), use_standardized=False, stochastic=True, pad=True)
            elif letter == 'T':
                unseen_dataset = TCOF('dev', args.n_seconds, args.downsampling, use_standardized=False, stochastic=True, pad=False)
            else:
                raise NotImplementedError(f"Character {letter} is not recognized as a dataset letter")
            val_datasets = append_datasets_val(val_datasets, unseen_dataset, letter)

    train_datasets = ClassConcatDataset(train_datasets)

    return train_datasets, val_datasets
