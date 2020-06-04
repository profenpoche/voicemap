#!/bin/bash

# 04/06/2020
# Runner for the training of voicemap - version 1
# Contains the hyperparameters (you can change them here)

## Logs
# The beginning and the end of the training is logged in /var/log/syslog
# The complete logs of the trainings can be found in voicemap/logs/training.log
# You can type `tail -f logs/training.log` to see the logs in live

path=~/"voicemap/"

logger Voicemap: Beginning of the training...

nohup python3.7 ${path}experiments/train.py \
	--model resnet \
	--dim 1  \
	--lr 0.002 \
	--weight-decay 0.05 \
	--momentum 0.9 \
	--filters 32 \
	--batch-size 64 \
	--n-seconds 3 \
	--spectrogram False \
	--window-length 0.02 \
	--window-hop 0.01 \
	--downsampling 4 \
	--epochs 1 \
	>> ${path}logs/training.log && logger Voicemap: Training ended &
