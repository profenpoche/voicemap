import torch
import argparse

from voicemap.utils import setup_dirs
from voicemap.train import train

###############################
#            Main             #
###############################

if __name__ == "__main__":

    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--model-path', type=str, help='Saved model to load to continue training')
    parser.add_argument('--dim', type=int)
    parser.add_argument('--lr', type=float, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--filters', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--n-seconds', type=float)
    parser.add_argument('--downsampling', type=int)
    parser.add_argument('--use-standardized', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Whether or not to use standardized data')
    parser.add_argument('--spectrogram', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Whether or not to use raw waveform or a spectogram as inputs.')
    parser.add_argument('--precompute-spect', type=lambda x: x.lower()[0] == 't', default=True,
                        help='Whether or not to calculate spectrograms on the fly from raw audio.')
    parser.add_argument('--window-length', type=float, help='STFT window length in seconds.')
    parser.add_argument('--window-hop', type=float, help='STFT window hop in seconds.')
    parser.add_argument('--n_t', type=int, default=0, help='Number of SpecAugment time masks.')
    parser.add_argument('--T', type=int, help='Maximum size of time masks.')
    parser.add_argument('--n_f', type=int, default=0, help='Number of SpecAugment frequency masks.')
    parser.add_argument('--F', type=int, help='Maximum size of frequency masks.')
    args = parser.parse_args()


    setup_dirs()
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    train(args, train_datasets_letters="LCT", val_datasets_letters="LSCT")
