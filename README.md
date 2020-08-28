# voicemap

This repository contains code to build deep learning models to identify
different speakers based on audio samples containing their voice.

The eventual aim is for this repository to become a pip-installable
python package for quickly and easily performing speaker identification
related tasks.

**This tensorflow/Keras/python2.7 branch is discontinued. Work is
continuing on the pytorch-python-3.6 branch which will become the
master branch.**

## Instructions
### Requirements
Make a new virtualenv and install requirements from `requirements.txt`
with the following command.
```
pip install -r requirements.txt
```
This project was written in Python 2.7.12 so I cannot guarantee it works
on any other version.

### Data
Get training data here: http://www.openslr.org/12
- train-clean-100.tar.gz
- train-clean-360.tar.gz
- dev-clean.tar.gz

Place the unzipped training data into the `data/` folder so the file
structure is as follows:
```
data/
    LibriSpeech/
        dev-clean/
        train-clean-100/
        train-clean-360/
        SPEAKERS.TXT
```

Please use the `SPEAKERS.TXT` supplied in the repo as I've made a few
corrections to the one found at openslr.org.

**Change the path of the data folder in the config.py file**

### Pretrained models

You can download pretrained models. Put them into the models/ folder so you don't have to change the path for the preset prediction.
```
cd models
wget https://mathia.education/voicemap/best_model.pt
wget https://mathia.education/voicemap/best_model_without_spec.pt
```

### Run tests

This requires the LibriSpeech data.
```
python -m unittest tests.tests
```

## Contents
### voicemap
This package contains re-usable code for defining network architectures,
interacting with datasets and many utility functions.

### experiments
This package contains experiments in the form of python scripts.

### notebooks
This folder contains Jupyter notebooks used for interactive
visualisation and analysis.


## Usage

### Prediction

You can use a saved model (in the folder models/) to predict the speakers of audios.
Two models are available : best_model.pt and best_model_without_spec.pt. The data used for the example for the prediction must be downloaded. You can launch prediction through the console with bin/predict.sh (you have to precise some parameters of the model) or (recommended) through the Jupyter notebook "Prediction".

Just run the command below to run prediction with a given model and given audio files
```
bin/predict.sh
```

### Speakers standardization
For training, you can toggle the 'use_standardized' parameter to keep only a given amount of samples per speaker.
An example of the standardization is given in the "Optimization" notebook. You can also see the notebook "Standardize_speakers" to execute the function to standardize speakers.

The standardization generates other csv files for the datasets. In these files every speaker have a fixed number of samples.

### Log the best models
Best models parameters are logged manually into the logs/models.csv. Thanks to the "Model_stats" notebook you can save models and log them into this log file.

### Parameters optimization
Thanks to the "Optimization" notebook, you can do hyperparameters optimization with Optuna. The logs are sent to Neptune. You will have to create an account and paste the Neptune API key into the notebook to use it.
