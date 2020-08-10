#!/bin/bash

# 04/06/2020
# Runner for the training of voicemap - version 1
# Contains the hyperparameters (you can change them here)

## Logs

# The beginning and the end of the training is logged in /var/log/syslog
# Type `syslog` to see them in live
# The complete logs of the trainings can be found in voicemap/logs/training.log
# Type `trainlog` to see them in live

path=~/"voicemap/"

logger Voicemap: Beginning of the training...
echo -e "\n\n`date` : Beginning of the training...\n\n" >> ${path}logs/training.log

## Comments about the hyperparams

# --model resnet \ # resnet ou baseline
# --model-path \ # path of the model to load before training
# --dim 1  \
# --lr 0.01 \ # Initial learning rate
# --weight-decay 0.05 \ # Prevents the weights from growing too large
# --momentum 0.9 \ # To avoid local extrema
# --filters 32 \
# --batch-size 32 \ # Number of samples processed before the model is updated
# --n-seconds 3 \
# --spectrogram False \ # Whether or not to use raw waveform or a spectogram as inputs
# --precompute-spect True \ # Whether or not to calculate spectrograms on the fly from raw audio. Used only if --spectrogram is True
# --n_t 0 \ # Number of SpecAugment time masks
# # --T \ # Maximum size of time masks
# --n_f 0 \ # Number of SpecAugment frequency masks
# # --F \ # Maximum size of frequency masks
# --window-length 0.02 \ # STFT window length in seconds
# --window-hop 0.01 \ # STFT window hop in seconds
# --downsampling 4 \
# --epochs 10 \ # Number of times the entire dataset passes through the NN

nohup python3.7 ${path}experiments/train.py \
    --model resnet \
	--dim 1 \
	--lr 0.001 \
	--weight-decay 0.01 \
	--momentum 0.9 \
	--filters 128 \
	--batch-size 1500 \
	--n-seconds 3 \
	--spectrogram True \
	--precompute-spect True \
	--use-standardized True \
	--n_t 0 \
	\
	--n_f 0 \
	\
	--window-length 0.02 \
	--window-hop 0.01 \
	--downsampling 4 \
	--epochs 50 \
	>> ${path}logs/training.log && logger Voicemap: Training ended &
	
	
