""" TODO : This file is obsolete. Have to put it in the voicemap folder as a library.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
import argparse
from olympic.callbacks import CSVLogger, Evaluate, ReduceLROnPlateau, ModelCheckpoint
from olympic import fit
from typing import Union, List

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, ClassConcatDataset, SpectrogramDataset, DatasetFolder, PredictionAudioDataset
from voicemap.models import ResidualClassifier, BaselineClassifier
from voicemap.utils import whiten, setup_dirs
from voicemap.eval import VerificationMetrics
from voicemap.augment import SpecAugment
from config import PATH, DATA_PATH
from voicemap.train import TrainingArgs, calculate_in_channels

from olympic.metrics import NAMED_METRICS
from voicemap.eval import unseen_speakers_evaluation

import soundfile as sf
from pydub import AudioSegment
import os
import sys
import time
start_time = time.time()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#############
# Datasets  #
#############

def generateDataloaders(args):
    test_size = 0.0005
    test_size_unseen = 0.01

    librispeech_subsets = ['train-clean-100', 'train-clean-360', 'train-other-500']
    unseen_subset = 'dev-clean'

    if args.spectrogram:
        librispeech = SpectrogramDataset(
            LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, stochastic=True, pad=False),
            normalisation='global',
            window_length=args.window_length,
            window_hop=args.window_hop
        )
        librispeech_unseen = SpectrogramDataset(
            LibriSpeech(unseen_subset, args.n_seconds, args.downsampling, stochastic=True, pad=False),
            normalisation='global',
            window_length=args.window_length,
            window_hop=args.window_hop
        )
    else:
        librispeech = LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, stochastic=True, pad=False)
        librispeech_unseen = LibriSpeech(unseen_subset, args.n_seconds, args.downsampling, stochastic=True, pad=False)
    

    # num_classes of the trained model (in local we have less speakers)
    # num_classes = librispeech.num_classes
    dataset_sampling_rate = LibriSpeech.base_sampling_rate

    if args.local:
        # test subset fixed in local
        test_indices = [0, 1, 5, 10, 100, 121, 500, 800, 1000]
        test_indices_unseen = [0, 1, 5, 10, 100, 121, 500, 800, 1000]
    else:
        # Test subset randomized (all dataset necessary)
        indices = range(len(librispeech))
        indices_unseen = range(len(librispeech_unseen))
        _, test_indices, _, _ = train_test_split(
            indices,
            indices,
            test_size=test_size,
            # stratify=speaker_ids
        )
        _, test_indices_unseen, _, _ = train_test_split(
            indices_unseen,
            indices_unseen,
            test_size=test_size_unseen,
            # stratify=speaker_ids
        )

    test_subset = torch.utils.data.Subset(librispeech, test_indices)
    test_subset_unseen = torch.utils.data.Subset(librispeech_unseen, test_indices_unseen)


    dataloader = DataLoader(test_subset, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=False)
    dataloader_unseen = DataLoader(test_subset_unseen, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=False)
    return {
        "dataloaders": [
            dataloader,
            dataloader_unseen
        ],
        "num_classes": args.num_speakers,
        "datasets_sampling_rate": dataset_sampling_rate
    }

##############################
# Loading of the saved model #
##############################
# Load the weights of the network
def load_model(model_path, filters, in_channels, num_classes, dim, device):
    if device == 'cpu':
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        assert torch.cuda.is_available()
        state_dict = torch.load(model_path)

    # Create the model with those weights
    model = ResidualClassifier(in_channels, filters, [2, 2, 2, 2], num_classes, dim=dim)
    model.to(device, dtype=torch.double)
    model.load_state_dict(state_dict=state_dict)
    print("Model loaded : "+model_path)
    model.eval()
    return model

#########################
# Prediction of subsets #
#########################

def predictSubset(dataloader, model):
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            print("Should be : ")
            print(y)
            y_pred = model(x)
            # Show the values predicted for each class
            values, indices = torch.max(torch.transpose(y_pred, 0, 1), 0)
            print(values)
            print("Predicted : ")
            print(indices)
        v = NAMED_METRICS['accuracy'](y, y_pred)
        print("accuracy : "+str(v*100)+"%")


###############################
# Prediction of an audio file #
###############################
# getitem from the dataset (ie audio array) and predict the speaker with the model
# Print the N_SPEAKERS_EVALUATION speakers it is most confident of
def predictAudioFile(predictionDataloader, model, n_first_spk=7, spectrogram=True, device='cuda'):

    if spectrogram:
        def prepare_batch(batch):
            # Normalise inputs
            # Move to GPU and convert targets to int
            x, y = batch
            return x.double().cuda(), y.long().cuda()
    else:
        def prepare_batch(batch):
            # Normalise inputs
            # Move to GPU if possible and convert targets to int
            x, y = batch
            if (device == 'cpu'):
                return whiten(x), y.long()
            else:
                return whiten(x).cuda(), y.long().cuda()


    with torch.no_grad():
        print("\n-------------- Prediction of an audio : " + predictionDataloader.dataset.df['filepath'][0])
        # try:
        for batch in predictionDataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)
        # except NameError:
        #     raise NameError
        # except:
        #     e = sys.exc_info()[0]
        #     print(e)
        values = y_pred.data[0].tolist()
        most_confident_speakers = []
        speakers_confidence = []
        for _ in range(n_first_spk):
            index = values.index(max(values))
            value = values[index]
            values[index] = 0
            speakers_confidence.append(value)
            most_confident_speakers.append(index)
        print("Predicted : ")
        print("\t\t\t".join(map(lambda x: str(x), most_confident_speakers)))
        print("\t".join(map(lambda x: str(x), speakers_confidence)))
        return y_pred.data[0].tolist(), most_confident_speakers

###############################
# Prediction - speaker association #
###############################
# Ensures that the predicted speaker was not associated to another audio
#
# Compare the recognized speakers for the current audio and the speakers already encountered with the previous audios
# If a speaker recognized was already encountered and we are confident in it, remove the possibility
def chooseSpeakerToKeep(speakers_recognized, speakers_trusted, confidences):
    if len(speakers_recognized) > 0:
        most_confident_speaker = speakers_recognized.pop(0)
    else:
        raise RuntimeError

    if most_confident_speaker in speakers_trusted:
        print("The speaker recognized was already attached for a previous person")
        return chooseSpeakerToKeep(speakers_recognized, speakers_trusted, confidences)
    else:
        return most_confident_speaker

# Return the speaker to associate with the audio
def identifySpeaker(speakers, predictionDataloader, model, n_first_spk=7, spectrogram=True, device='cuda'):
    speaker_identified = -1
    while speaker_identified == -1:
        start_time = time.time()
        # try:
        values, speakers_recognized = predictAudioFile(predictionDataloader, model, n_first_spk, spectrogram, device)
        speaker_identified = chooseSpeakerToKeep(speakers_recognized, speakers, values)
        print("--- %s seconds ---" % (time.time() - start_time))
    return speaker_identified

###############################
#  Util functions             #
###############################
def addFileToPredictionDataset(filepath, predictionDataset):
    predictionDataset.append(filepath)
    return len(predictionDataset) - 1

def buildPredictionDataloader(filename, args):
    predictionDataset = PredictionAudioDataset(filename, args.n_seconds, args.downsampling)
    if args.spectrogram:
        predictionDataset = SpectrogramDataset(predictionDataset, 'global', args.window_length, args.window_hop)
    return DataLoader(predictionDataset)

###############################
#            Main             #
###############################

if __name__ == "__main__":

    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="saved_models/lr=0.001__acc=0.81.pt")
    parser.add_argument('--audio-path', type=str, default="data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
    # parser.add_argument('--audio-path', type=str, default="/home/iribarnesy/Documents/Info/Stage_Pep/voicemap-pep2/voicemap/data/LibriSpeech/dev-clean/174/168635/174-168635-0005.flac")
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--filters', type=int)
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-seconds', type=float, default=3)
    parser.add_argument('--downsampling', type=int, default=4)
    parser.add_argument('--spectrogram', type=lambda x: x.lower()[0] == 't', default=False,
                        help='Whether or not to use raw waveform or a spectogram as inputs.')
    parser.add_argument('--window-length', type=float, help='STFT window length in seconds.', default=0.02)
    parser.add_argument('--window-hop', type=float, help='STFT window hop in seconds.', default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--local',  type=lambda x: x.lower()[0] == 't', default=False)
    args = parser.parse_args()


    in_channels = calculate_in_channels(args)
    N_SPEAKERS_EVALUATION = 7
    
    ##############
    # Prediction #
    ##############
    print("--- %s seconds ---" % (time.time() - start_time))

    print("\nBeginning of the activity\n")

    print(bcolors.HEADER + "#########################################") 
    print("Initialisation phase :")
    print("#########################################" + bcolors.ENDC) 
    n_members = 6
    auto_prepare = int(input("Do you want to auto prepare "+str(n_members)+" speakers ? (0/1) : "))
    if (auto_prepare):
        audio_paths = [
            DATA_PATH + "/testPrediction/activity_SId/1-10.flac", 
            DATA_PATH + "/testPrediction/activity_SId/2-18.flac", 
            DATA_PATH + "/testPrediction/activity_SId/3-18.flac", 
            DATA_PATH + "/testPrediction/activity_SId/6-4.flac",
            DATA_PATH + "/testPrediction/activity_SId/7-2.flac",
            DATA_PATH + "/testPrediction/activity_SId/8-4.flac"
            # DATA_PATH + "/testPrediction/activity_SId/4-1.flac", 
            # DATA_PATH + "/testPrediction/activity_SId/5-1.flac"
            ]
    else:
        n_members = int(input("Please enter the number of speakers in the activity : "))

    # Load the model
    model = load_model(args.model_path, args.filters, in_channels, args.num_speakers, args.dim, args.device)
    # Initialize the list of identified and trusted speakers
    speakers = [] 

    # First step is to associate speakers with audio file foreach member of the group
    for m in range(n_members):
        if auto_prepare:
            audio_path = audio_paths[m]
        else:
            audio_path = str(input("Please enter the audio path for the voice of the speaker n°"+str(m)+" : "))
        
        # It is necessary to put the audio in a dataloader before prediction
        predictionDataloader = buildPredictionDataloader(audio_path, args)
        
        # predict the speaker (and validate him compared to the previous speakers)
        speaker_identified = identifySpeaker(speakers, predictionDataloader, model, N_SPEAKERS_EVALUATION, args.spectrogram, args.device)
        speakers.append(speaker_identified)
        
    print(bcolors.OKBLUE + "\nSpeakers are : ")
    print(speakers)
    print(bcolors.ENDC)

    # Then we can loop and ask for an unlimited number of audio file
    # Foreach audio we can link the most confident speaker among the 'speakers' list
    print(bcolors.HEADER + "#########################################") 
    print("Prediction phase :")
    print("#########################################" + bcolors.ENDC) 

    activity_can_continue = True
    while activity_can_continue:
        try:
            audio_path = str(input("Please enter the audio path for the speaker to identify : "))
            predictionDataloader = buildPredictionDataloader(audio_path, args)
            values, indices = predictAudioFile(predictionDataloader, model, args.spectrogram, args.device)
            print("Confidence for each speaker previously identified : ")
            filtered_list = [values[i] for i in speakers]
            print(filtered_list)
            print(bcolors.BOLD + "====== Speaker identified is : "+str(filtered_list.index(max(filtered_list))) + bcolors.ENDC)
        except NameError as error:
            print(error)
            activity_can_continue = bool(input("NameError. Press Enter to leave "))
        except RuntimeError as error:
            print(error)
            activity_can_continue = bool(input("RuntimeError. Press Enter to leave "))

    # print("\n-------------- Prediction of a test subset : librispeech")
    # predictSubset(dataloader, model)

    # print("\n-------------- Prediction of a test subset : librispeech (unseen)")
    # predictSubset(dataloader_unseen, model)
    # print("See below (with gpu) for the accuracy for unseen speakers")

    # if args.device == 'cuda':
    #     print("\n-------------- Prediction of a test subset : librispeech (unseen). Better analysis")
    #     for k, v in unseen_speakers_evaluation(model, librispeech_unseen, 250).items():
    #         print(k,v)


