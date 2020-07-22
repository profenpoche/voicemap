#!/bin/bash

# 10/06/2020
# Runner for prediction with voicemap - version 1
# You have to precise some hyparameters of the model you are predicting with (you can change them here)

path=~/"voicemap/"

echo -e "\nInfo: Change the model and the audio file used in bin/predict.sh\n"

## Comments about the hyperparams

# --model-path "saved_models/lr=0.0001__epoch=35__acc=0.865.pt" \
# --audio-path "data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac" \
# --dim 1 \
# --filters \
# --batch-size 64 \ # for the reading of the datasets
# --n-seconds 3 \
# --downsampling 4 \
# --window-length 0.1 \ # STFT window length in seconds
# --device cuda 

python3.7 ${path}experiments/predict.py \
	--model-path "saved_models/gaixoa.pt" \
	--audio-path "data/testPrediction/pob.flac" \
	--filters 128 \
	--num-speakers 1291 \
	--spectrogram False \
	
	
