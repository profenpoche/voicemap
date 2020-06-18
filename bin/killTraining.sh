#!/bin/bash

# 04/06/2020
# Killer for the training of voicemap - version 1
# Actually, it kills the python3.7 process higher in the tree

sudo kill `ps -ef | grep python3.7 | cut -f 2 -d" " | head -n 1`
logger Python3.7 process killed
