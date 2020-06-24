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

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, ClassConcatDataset, SpectrogramDataset, DatasetFolder
from voicemap.models import ResidualClassifier, BaselineClassifier
from voicemap.utils import whiten, setup_dirs
from voicemap.eval import VerificationMetrics
from voicemap.augment import SpecAugment
from config import PATH, DATA_PATH

from olympic.metrics import NAMED_METRICS
from voicemap.eval import unseen_speakers_evaluation

import soundfile as sf
from pydub import AudioSegment
import os

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

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="saved_models/lr=0.001__acc=0.81.pt")
parser.add_argument('--audio-path', type=str, default="data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
# parser.add_argument('--audio-path', type=str, default="/home/iribarnesy/Documents/Info/Stage_Pep/voicemap-pep2/voicemap/data/LibriSpeech/dev-clean/174/168635/174-168635-0005.flac")
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--filters', type=int)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--n-seconds', type=float, default=3)
parser.add_argument('--downsampling', type=int, default=4)
parser.add_argument('--window-length', type=float, help='STFT window length in seconds.', default=0.1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--local', type=bool, default=False)
args = parser.parse_args()

#############
# Constants #
#############
# Device must be 'cpu' if no gpu found
device = torch.device(args.device)
in_channels = 1
test_size = 0.0005
test_size_unseen = 0.01
num_classes = 1291

N_SPEAKERS_EVALUATION = 7

#############
# Datasets  #
#############

def generateDataloaders():
    librispeech_subsets = ['train-clean-100', 'train-clean-360', 'train-other-500']
    unseen_subset = 'dev-clean'

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
        "num_classes": num_classes,
        "datasets_sampling_rate": dataset_sampling_rate
    }

def prepare_batch(batch):
    # Normalise inputs
    # Move to GPU if possible and convert targets to int
    x, y = batch
    if (args.device == 'cpu'):
        return whiten(x), y.long()
    else:
        return whiten(x).cuda(), y.long().cuda()

##############################
# Loading of the saved model #
##############################
# Load the weights of the network
if args.device == 'cpu':
    state_dict = torch.load(args.model_path, map_location='cpu')
else:
    assert torch.cuda.is_available()
    state_dict = torch.load(args.model_path)

# Create the model with those weights
model = ResidualClassifier(in_channels, args.filters, [2, 2, 2, 2], num_classes, dim=args.dim)
model.to(device, dtype=torch.double)
model.load_state_dict(state_dict=state_dict)
print("Model loaded : "+args.model_path)
model.eval()

#########################
# Prediction of subsets #
#########################

def predictSubset(dataloader):
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

################################
# Extraction of an audio file #
################################
# Conversion if necessary (only .wav -> .flac)
def getAudioFromFile(audio_path, dataset_sampling_rate, n_seconds, downsampling, device):
    filename, extension = os.path.splitext(audio_path)
    if extension == '.wav' :
        audio_path = filename + ".flac"
        song = AudioSegment.from_wav(audio_path)
        song.export(audio_path,format = "flac")
    elif extension == '.flac':
        pass
    else:
        raise RuntimeError

    # Read the .flac file
    instance, samplerate = sf.read(audio_path)
    # Reduce its size
    fragment_length = int(n_seconds * dataset_sampling_rate)
    fragment_start_index = 0
    if args.n_seconds is not None:
        instance = instance[fragment_start_index:fragment_start_index+fragment_length]

    # Extract the first channel (necessary if the file is double-channel, not usual)
    if len(instance.shape) > 1:
        instance = np.transpose(instance)
        instance = instance[0]
        instance = np.transpose(instance)

    instance = instance[np.newaxis, np.newaxis, ::downsampling]
    instance = torch.from_numpy(instance).to(device)
    return instance

###############################
# Prediction of an audio file #
###############################
def predictAudioFile(audio_path, model, sampling_rate, n_seconds, downsampling, device):
    with torch.no_grad():
        print("\n-------------- Prediction of an audio : " + audio_path)
        instance = getAudioFromFile(audio_path, sampling_rate, n_seconds, downsampling, device)
        y_pred = model(instance)

        values = y_pred.data[0].tolist()
        most_confident_speakers = []
        speakers_confidence = []
        for _ in range(N_SPEAKERS_EVALUATION):
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
# Prediction of an audio file #
###############################
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

def identifySpeaker(speakers, audio_path, model, sampling_rate, n_seconds, downsampling, device):
    speaker_identified = -1
    while speaker_identified == -1:
        start_time = time.time()
        values, speakers_recognized = predictAudioFile(audio_path, model, sampling_rate, n_seconds, downsampling, device)
        try:
            speaker_identified = chooseSpeakerToKeep(speakers_recognized, speakers, values)
        except RuntimeError as error:
            print(error)
            print("The voice is too similar to precedent voices. Please retry to speak...")
            audio_path = str(input("Please enter another audio path for the same speaker : "))
            speaker_identified = identifySpeaker(speakers, audio_path, model, sampling_rate, n_seconds, downsampling, device)
        print("--- %s seconds ---" % (time.time() - start_time))
    return speaker_identified


###############################
#            Main             #
###############################
print("--- %s seconds ---" % (time.time() - start_time))

print("\nBeginning of the activity\n")

print(bcolors.HEADER + "#########################################") 
print("Intialisation phase :")
print("#########################################" + bcolors.ENDC) 
n_members = 5
auto_prepare = int(input("Do you want to auto prepare "+str(n_members)+" speakers ? (0/1) : "))
if (auto_prepare):
    audio_paths = [
        "data/testPrediction/activity_SId/1-10.flac", 
        "data/testPrediction/activity_SId/2-18.flac", 
        "data/testPrediction/activity_SId/3-18.flac", 
        "data/testPrediction/activity_SId/4-1.flac", 
        "data/testPrediction/activity_SId/5-1.flac"
        ]
else:
    n_members = int(input("Please enter the number of speakers in the activity : "))

speakers = []
for m in range(n_members):
    if auto_prepare:
        audio_path = audio_paths[m]
    else:
        audio_path = str(input("Please enter the audio path for the voice of the speaker n°"+str(m)+" : "))
    speaker_identified = identifySpeaker(speakers, audio_path, model, LibriSpeech.base_sampling_rate, args.n_seconds, args.downsampling, device)
    speakers.append(speaker_identified)
print(bcolors.OKBLUE + "\nSpeakers are : ")
print(speakers)
print(bcolors.ENDC)

print(bcolors.HEADER + "#########################################") 
print("Prediction phase :")
print("#########################################" + bcolors.ENDC) 

activity_can_continue = True
while activity_can_continue:
    try:
        audio_path = str(input("Please enter the audio path for the speaker to identify : "))
        values, indices = predictAudioFile(audio_path, model, LibriSpeech.base_sampling_rate, args.n_seconds, args.downsampling, device)
        print("Confidence for each speaker previously identified : ")
        filtered_list = [values[i] for i in speakers]
        print(filtered_list)
        print(bcolors.BOLD + "====== Speaker identified is : "+str(filtered_list.index(max(filtered_list))) + bcolors.ENDC)
    except RuntimeError as error:
        print(error)
        activity_can_continue = bool(input("RuntimeError. Press Enter to leave "))

# print("\n-------------- Prediction of a test subset : librispeech")
# predictSubset(dataloader)

# print("\n-------------- Prediction of a test subset : librispeech (unseen)")
# predictSubset(dataloader_unseen)
# print("See below (with gpu) for the accuracy for unseen speakers")

# if args.device == 'cuda':
#     print("\n-------------- Prediction of a test subset : librispeech (unseen). Better analysis")
#     for k, v in unseen_speakers_evaluation(model, librispeech_unseen, 250).items():
#         print(k,v)

