# voicemap

This repository contains code to build deep learning models to identify
different speakers based on audio samples containing their voice.

The eventual aim is for this repository to become a pip-installable
python package for quickly and easily performing speaker identification
related tasks.

**French only : Pour les francophones, un document plus détaillé sur le fonctionnement de voicemap est accessible ici : https://drive.google.com/file/d/1tt1qUUiNg3PivoT4hgLydDRQoDjSdC3m/view?usp=sharing**

## Instructions
### Requirements
Make a new virtualenv and install requirements from `requirements.txt`
with the following command.
```
pip install -r requirements.txt
```

### Pretrained models

You can download pretrained models. Put them into the models/ folder so you don't have to change the path for the preset prediction.
```
cd models
wget https://mathia.education/voicemap/best_model.pt
wget https://mathia.education/voicemap/best_model_without_spec.pt
```

### Data
Four different datasets are ready to be used :
- LibriSpeech
- Speakers in the Wild
- Common Voice
- TCOF

For each dataset you have to download it and place it in the data/ folder.
You must have a training set and a validation set, the filepaths must be listed in a .csv file.
The manner to read respectively each dataset is described into the voicemap/datasets/\<dataset\>.py file.

Info : The usage put in parenthesis for the subsets are indicative. You can change the usage if you want.

**Change the path of the data folder in the config.py file**

#### LibriSpeech
Get training data here: http://www.openslr.org/12
- train-clean-100.tar.gz (used for training)
- train-clean-360.tar.gz (used for training)
- dev-clean.tar.gz (used for validation)

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

Please use the `SPEAKERS.TXT` supplied in the repo as the one found at openslr.org has been modified.

#### SitW
Get training data here: http://www.speech.sri.com/projects/sitw/
- sitw_database.v4.tar.gz

Place the unzipped training data into the `data/` folder so the file
structure is as follows:
```
data/
    sitw/
        dev/ (used for training)
        eval/ (used for validation)
```

#### Common Voice
Get training data here: https://commonvoice.mozilla.org/en/datasets

For instance, for the corpus containing french audios :
- fr.tar.gz

Place the unzipped training data into the `data/` folder so the file
structure is as follows:
```
data/
    CommonVoice/
        fr/
            validated.tsv (unused)
            train.tsv (training set)
            test.tsv (validation set)
            dev.tsv
            other.tsv
```

#### TCOF, French dataset
Get training data here: https://www.ortolang.fr/market/corpora/tcof?path=%2FCorpus
For instance, for the corpus containing french kids :
- Enfants.zip 

Place the unzipped training data into the `data/` folder so the file
structure is as follows:
```
data/
    TCOF/
        TCOF_train.csv (train)
        TCOF_test.csv
        TCOF_dev.csv (val)
        Enfants/
```

**The .csv file is not downloadable over the website, you will have to generate it.**

You can use the notebook "Imports and transforms datasets" notebook and the "importTCOF.py" script to generate the .csv file listing the filepaths. The script will separate the different speakers over the audios thanks to the timecodes written in the .xml transcript files.

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

You can use a saved model (in the models/ folder) to predict the speakers of audios.
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
