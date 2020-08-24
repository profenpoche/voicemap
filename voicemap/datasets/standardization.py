"""
Standardization of the speakers
Date : 24/08/2020
Refactorization of the 'Standardize_speakers' notebook

Load all the datasets from the global csv. 
Select a fixed number of samples for all the speakers (ex : 60 sec => 3s x 20 samples). 
You will have to load the datasets with 'use_standardized=True' to use the standardization saved for each dataset 
(ex : TCOF_train.csv => TCOF_train_standardized.csv)

"""
# useful imports
import pandas as pd
from tqdm.autonotebook import tqdm
from typing import List

# voicemap imports
from voicemap.datasets import LibriSpeech, SpeakersInTheWild, CommonVoice, TCOF, ClassConcatDataset, SpectrogramDataset, DatasetFolder
from voicemap.datasets.datasets import letter_to_datafolder_dict
from config import PATH, DATA_PATH


def load_data(subset: str):
    """ Load data from the global csv gathering all the samples
    """
    df_train = pd.read_csv(f"{DATA_PATH}/train_durations_per_speaker.csv")
    df_test = pd.read_csv(f"{DATA_PATH}/val_durations_per_speaker.csv")
    df_global = pd.read_csv(f"{DATA_PATH}/global_durations_per_speaker.csv")
    if (subset == "train"):
        df = df_train
    elif (subset == "val"):
        df = df_test
    else:
        df = df_global
    return df

def prepare_data(base_df, n_seconds_min=3):
    """ Prepares data, removes too short samples.
    Groups the speakers by id and counts their number of samples
    """
    # Remove too short samples
    source_df = base_df.loc[base_df['seconds'] > n_seconds_min]
    # Group speakers duplicated by id
    df = source_df.loc[:, ['speaker_id', 'dataset_name']]
    df = df.set_index('speaker_id')
    df = df.loc[~df.index.duplicated(keep='first')]
    dfGrouped = source_df.groupby(['speaker_id']).sum()
    # Count the number of samples for each speaker
    dfCountAudio = source_df.groupby(['speaker_id']).count().filepath
    speakers_duration = dfGrouped.join(df)
    speakers_duration = speakers_duration.join(dfCountAudio)
    speakers_duration = speakers_duration.rename(columns={'filepath': 'n_samples'})
    return source_df, speakers_duration


def standardize_samples_for_speakers(source_df, speakers_duration, n_samples_min, verbose=True):
    """ Returns all the samples of the speakers who speak enough (ie. n_samples >= n_samples_min).
    Sorts the samples by duration decreasing.
    """
    speakers_duration = speakers_duration.loc[speakers_duration.n_samples >= n_samples_min]
    if verbose:
        print(f"{len(speakers_duration)} speakers remain")

    # get the samples > n_seconds for the remaining speakers
    samples_remaining_speakers = pd.merge(source_df, speakers_duration, on="speaker_id")
    # sort the samples by speakers and duration
    samples_remaining_speakers = samples_remaining_speakers.sort_values(['speaker_id', 'seconds_x'], ascending=[True, True])
    if verbose:
        print(f"{len(samples_remaining_speakers.loc[samples_remaining_speakers['dataset_name_x'] == 'LibriSpeech'])} samples remain in LibriSpeech")
        print(f"{len(samples_remaining_speakers.loc[samples_remaining_speakers['dataset_name_x'] == 'sitw'])} samples remain in sitw")
        print(f"{len(samples_remaining_speakers.loc[samples_remaining_speakers['dataset_name_x'] == 'CommonVoice'])} samples remain in CommonVoice")
        print(f"{len(samples_remaining_speakers.loc[samples_remaining_speakers['dataset_name_x'] == 'TCOF'])} samples remain in TCOF")
    return samples_remaining_speakers, speakers_duration

def getSpeakerFirstSamples(df_samples, speaker_id, n_samples):
    """ Extract the 'n_samples' first samples for a given speaker
    """
    df_res = df_samples.loc[df_samples['speaker_id'] == speaker_id].head(n_samples)
    return df_res    

def filter_samples(samples_remaining_speakers, speakers_duration, n_samples_min, verbose=True):
    """ for each speaker add his 'n_samples' longest samples to the 'filtered_samples' dataframe
    """
    filtered_samples = pd.DataFrame()
    if verbose:
        print(f"Gathering {n_samples_min} samples for the {len(speakers_duration)} speakers...")
    for speaker_id in tqdm(speakers_duration.index):
        speaker_samples = getSpeakerFirstSamples(samples_remaining_speakers, speaker_id, n_samples_min)
        filtered_samples = pd.concat([filtered_samples, speaker_samples], sort=False)
    filtered_samples = filtered_samples.loc[:, ['speaker_id', 'filepath', 'seconds_x', 'dataset_name_x']]
    if verbose:
        print(f"Finished. {len(filtered_samples)} samples have been indexed.")
    return filtered_samples

def generate_standardized_csv(datasets_to_generate_csv_for: List[str], filtered_samples, subset, verbose=True):
    """ Generate standardized csv for datasets listed in 'datasets_to_generate_csv_for'
    """
    for dataset_name in datasets_to_generate_csv_for:
        dataset_samples = filtered_samples.loc[filtered_samples['dataset_name_x'] == dataset_name]
        dataset_samples = dataset_samples.rename(columns={'seconds_x': 'seconds'})
        dataset_samples = dataset_samples.loc[:, ['speaker_id', 'filepath', 'seconds']]
        dataset_samples.to_csv(f"{DATA_PATH}/{dataset_name}/{dataset_name}_{subset}_standardized.csv", index=False)
        if verbose:
            print(f"Saved to '{DATA_PATH}/{dataset_name}/{dataset_name}_{subset}_standardized.csv'")

def standardize_speakers(subset, datasets_letters, n_samples_min, n_seconds_min=3, verbose=True):
    """ Main function to standardize speakers.
        Keep only the speakers with more than n_samples_min samples longer than n_seconds_min.
        Save the filepaths into a standardized csv for each dataset

        Args:
        subset: 'train', 'val' or 'global' for the subsets to load for each dataset.
        datasets_letters: datasets to standardize speakers for. Must be part of the letter_to_datafolder_dict in voicemap.datasets.datasets.
        n_samples_min: number of samples to keep for a speaker (eliminate speakers with less samples)
        n_seconds_min: minimum length of the samples to keep
    """
    datasets_to_generate = [letter_to_datafolder_dict[letter] for letter in datasets_letters]
    base_df = load_data(subset)
    source_df, speakers_duration = prepare_data(base_df, n_seconds_min)
    samples_remaining_speakers, speakers_duration = standardize_samples_for_speakers(source_df, speakers_duration, n_samples_min, verbose)
    filtered_samples = filter_samples(samples_remaining_speakers, speakers_duration, n_samples_min, verbose)
    generate_standardized_csv(datasets_to_generate, filtered_samples, subset, verbose)