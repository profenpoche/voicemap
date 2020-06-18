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

#############
# Datasets  #
#############
librispeech_subsets = ['train-clean-100', 'train-clean-360', 'train-other-500']
unseen_subset = 'dev-clean'

librispeech = LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, stochastic=True, pad=False)
librispeech_unseen = LibriSpeech(unseen_subset, args.n_seconds, args.downsampling, stochastic=True, pad=False)

# num_classes of the trained model (in local we have less speakers)
# num_classes = librispeech.num_classes
num_classes = 1291
base_sampling_rate = LibriSpeech.base_sampling_rate

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

#########################
# Prediction of subsets #
#########################
model.eval()
with torch.no_grad():
    print("\n-------------- Prediction of a test subset : librispeech")
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

    print("\n-------------- Prediction of a test subset : librispeech (unseen)")
    for batch in dataloader_unseen:
        x, y = prepare_batch(batch)
        print("Should be : ")
        print(y)
        y_pred = model(x)
        # Show the values predicted for each class
        values, indices = torch.max(torch.transpose(y_pred, 0, 1), 0)
        print(values)
        print("Predicted : ")
        print(indices)
    print("See below (with gpu) for the accuracy for unseen speakers")
    
    if args.device == 'cuda':
        print("\n-------------- Prediction of a test subset : librispeech (unseen). Better analysis")
        for k, v in unseen_speakers_evaluation(model, librispeech_unseen, 250).items():
            print(k,v)


################################
# Extraction of an audio file #
################################
# Conversion if necessary (only .wav -> .flac)
filename, extension = os.path.splitext(args.audio_path)
if extension == '.wav' :
    audio_path = filename + ".flac"
    song = AudioSegment.from_wav(args.audio_path)
    song.export(audio_path,format = "flac")
elif extension == '.flac':
    audio_path = args.audio_path
else:
    raise RuntimeError

# Read the .flac file
instance, samplerate = sf.read(audio_path)
# Reduce its size
fragment_length = int(args.n_seconds * base_sampling_rate)
fragment_start_index = 0
if args.n_seconds is not None:
    instance = instance[fragment_start_index:fragment_start_index+fragment_length]

# Extract the first channel (necessary if the file is double-channel, not usual)
if len(instance.shape) > 1:
    instance = np.transpose(instance)
    instance = instance[0]
    instance = np.transpose(instance)

instance = instance[np.newaxis, np.newaxis, ::4]
instance = torch.from_numpy(instance).to(device)

###############################
# Prediction of an audio file #
###############################
with torch.no_grad():
    print("\n-------------- Prediction of an audio : " + filename.split('/')[-1] + extension)
    y_pred = model(instance)
    values, indices = torch.max(torch.transpose(y_pred, 0, 1), 0)
    print(values)
    print("Predicted : ")
    print(indices)