#!/bin/bash

# 04/06/2020
# Killer for the training of voicemap - version 1
# Actually, it kills the python3.7 process higher in the tree

sudo kill `nvidia-smi -q -d PIDS | grep "Process ID" | cut -f 2 -d":" | head -n 1`
logger Python3.7 process killed
