#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################
# Import tool for TCOF dataset
# Speakers_id are evaluated in this version
# Look at the FIELDNAMES attribute (and export function) to choose the fields to export
########################################################
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import argparse
import os
import csv
import re
import subprocess
import unicodedata
import sys

from os import path
from xml.dom import minidom
from pydub import AudioSegment

FIELDNAMES = ['speaker_id', 'filepath', 'transcript', 'wav_filesize'] #['wav_filename', 'wav_filesize', 'transcript']
JOKER_PEOPLE = ["adulte", "autre enfant"] # must be lowercase
FILTER = [r"¤.*?¤",r"\*\*\*",r"\A\s*\Z",r"\A\s*\Z",r"\${1,3}"]

SAMPLE_RATE = 16000
MAX_SECS = 10
#remove quote
regex = r"""
    ^\"(.*)\"$
    """
    
    
#clean phrase for each sub list we have the regex pattern and the substitute
CLEANER = [[regex,"\\1"],[r"^\+ ",''],[r"\-( |\n)",' '],[r"\r|\n",''],[r" \+",' '],[r"&gt;",''],[r"&lt;",''],[r"/|#",''],[r" ?\* ?",' '],[r"\s{2,}",' '],[r"\r",'']]
    
# Recursive function to naviagate into the folders and files.
# Foreach .trs file found, process it
# speaker_index is a counter for the speakers, it is incremented each time a new speaker is found
def process_directory(dir_path, speaker_index=0, previous_speakers=None):
    print('directory: "%s"' % dir_path)
    
    outputList = []
    for sub_dir in os.listdir(dir_path):
        # print("new sub_dir", sub_dir, "prev spk", previous_speakers)
        if (os.path.isdir(os.path.join(dir_path, sub_dir))):
            updatedList, speaker_index, previous_speakers = process_directory(os.path.join(dir_path, sub_dir), speaker_index, previous_speakers)
            outputList.extend(updatedList)
        if os.path.isfile(os.path.join(dir_path, sub_dir)):
            if sub_dir.endswith('.trs'):
                updatedList, speaker_index, previous_speakers = process_trs_file(sub_dir,dir_path,speaker_index,previous_speakers)
                outputList.extend(updatedList)

    return outputList, speaker_index, previous_speakers

def export_csv(outputList,dir_path):    
    target_csv_template = os.path.join(dir_path, 'TCOF_{}.csv')
    with open(target_csv_template.format('train'), 'w', newline='', encoding='utf-8') as train_csv_file:  # 80%
        with open(target_csv_template.format('dev'), 'w',newline='' , encoding='utf-8') as dev_csv_file:  # 10%
            with open(target_csv_template.format('test'), 'w',newline='' , encoding='utf-8') as test_csv_file:  # 10%
                train_writer = csv.DictWriter(train_csv_file, fieldnames=FIELDNAMES)
                train_writer.writeheader()
                dev_writer = csv.DictWriter(dev_csv_file, fieldnames=FIELDNAMES)
                dev_writer.writeheader()
                test_writer = csv.DictWriter(test_csv_file, fieldnames=FIELDNAMES)
                test_writer.writeheader()
                print(outputList[0:5])
                for i, item in enumerate(outputList):
                    i_mod = i % 10
                    if i_mod == 0:
                        writer = test_writer
                    elif i_mod == 1:
                        writer = dev_writer
                    else:
                        writer = train_writer
                        #less then 3 digit wav length are corrupted
                    try:
                        if(int(item[2]) > 1200):
                            writer.writerow(dict(
                                speaker_id=item[3],
                                filepath=item[0],
                                transcript=item[1],
                                wav_filesize=item[2],
                            ))
                    except TypeError as error:
                        print(error)
                        print("bugging element is : ")
                        print(item)
                    except:
                        e = sys.exc_info()[0]
                        print(e)
                        print("bugging element is : ")
                        print(item)

               
def filter_row(text):
    for pattern in FILTER :
        if(re.search(pattern, text)) :
            return False
    return True
    

def clean_text(text):        
    for pattern in CLEANER :
        text = re.sub(pattern[0], pattern[1], text, 0, re.MULTILINE)
    return text                  


            
def process_trs_file(file,dir_path,speaker_index=-1,previous_speakers=None):
    print("This is the previous speaker", previous_speakers)
    print('Trs file "%s" with index %d - start processing...' % (file, speaker_index))
    outputList = []
    
    filePath = os.path.join(dir_path, file)
    trsDom = minidom.parse(filePath)    
    
    audioFilenameWe = os.path.splitext(file)[0]
    audioFilename = audioFilenameWe+'.wav'
    try :
        audioFile = AudioSegment.from_wav(os.path.join(dir_path, audioFilename))
        audioFile = audioFile.set_channels(1)
    except :
        return outputList, speaker_index, previous_speakers
    sections = trsDom.getElementsByTagName('Section')
    speakers = trsDom.getElementsByTagName('Speaker')

    # Use the previous speaker if it is the same
    if previous_speakers is not None:
        sameSpeakerThanPrevious = bool(re.match(rf"{previous_speakers['prefix']}[0-9]*{previous_speakers['suffix']}", audioFilenameWe))
        # print("previous", f"{previous_speakers['prefix']}[0-9]*{previous_speakers['suffix']}","current",audioFilenameWe)
        # print("Result", sameSpeakerThanPrevious)
        if (sameSpeakerThanPrevious):
            speakers_id = previous_speakers['speakers']
    # Prepare for a new speaker if the speaker is different
    if (previous_speakers is None or not sameSpeakerThanPrevious):
        speakers_id = {}
    print("Current speakers are : ")
    for speaker in speakers:
        speaker_spkID = speaker.attributes['id'].value
        speaker_name = speaker.attributes['name'].value
        print(f"speaker_spkID : {speaker_spkID} | speaker_name : {speaker_name}")
        # Sometimes the same person is identified with different id ("spkXX"), but we assume the name is always the same
        if (speaker_name.lower() not in JOKER_PEOPLE):
            # the speaker is a named person
            if (speaker_name in speakers_id.keys()):
                # so first, get his index
                uniquePersonIndex = speakers_id[speakers_id[speaker_name]] # Name => spkXX => int
                # Add this index for the spk identified in this file
                speakers_id[speaker_spkID] = uniquePersonIndex
            else:
                speakers_id[speaker_name] = speaker_spkID

        if (speaker_spkID not in speakers_id.keys()):
            speakers_id[speaker_spkID] = speaker_index
            speaker_index+=1

    audioPartId=0
    
    for section in sections :        
        turns=section.getElementsByTagName('Turn')      
        for turn in turns :
            turnEndTime = int(float(turn.attributes['endTime'].value)*1000)
            try:
                turnSpeaker = turn.attributes['speaker'].value
                if (len(turnSpeaker.split(" ")) > 1):
                    raise ValueError("Speaker must be one word, whereas '"+turnSpeaker+"' was found")
                # print("turn of ", turnSpeaker)
                rawTurnString = turn.toxml()
                #remove Event|Comment|Who tag
                regex = r"((<(Comment|Event|Who|Turn)[^(><)]+(>|/>))|</Turn>)"
                rawTurnString = re.sub(regex, '', rawTurnString, 0, re.MULTILINE)
                #print(rawTurnString)
                #find all sync line pack in a tuple with transcript
                syncs = re.findall(r"(<Sync[^(><)]+>)([^(><)]*)", rawTurnString, re.MULTILINE | re.DOTALL)            
                for sync in syncs :
                    cleanText = clean_text(sync[1])
                    if filter_row(cleanText):
                        syncStartTime = int(float(re.findall(r"time=\"(.*)\"", sync[0], re.MULTILINE)[0])*1000)
                        # print(syncStartTime)
                        syncEndTime = 0
                        # print((len(syncs)-1))
                        # print(syncs.index(sync))
                        if (len(syncs)-1) == syncs.index(sync) :
                            syncEndTime = turnEndTime
                        else :
                            nextSync = syncs[syncs.index(sync)+1]
                            syncEndTime = int(float(re.findall(r"time=\"(.*)\"", nextSync[0], re.MULTILINE)[0])*1000)
                        # print('starttime')
                        # print(syncStartTime)
                        # print('endtime')
                        # print(syncEndTime)
                        audioPart = audioFile[syncStartTime:syncEndTime]
                        audioPart = audioPart.set_frame_rate(SAMPLE_RATE)
                        audioPartFilename = audioFilenameWe + '_' + str(audioPartId) + '.wav'
                        cleanText=clean_label(cleanText);
                        if PARAMS.r:
                            audioPart.export(os.path.join(CORPUS_DIR, audioPartFilename), format="wav",parameters=["-r", str(SAMPLE_RATE)])
                            if check_file_length(os.path.join(CORPUS_DIR, audioPartFilename),cleanText):                   
                                outputList.append([audioPartFilename,cleanText,os.path.getsize(os.path.join(CORPUS_DIR, audioPartFilename))])
                        else:
                            audioPart.export(os.path.join(dir_path, audioPartFilename), format="wav",parameters=["-r", str(SAMPLE_RATE)])
                            if(check_file_length(os.path.join(dir_path, audioPartFilename),cleanText)): 
                                # outputList.append([os.path.join(dir_path, audioPartFilename),cleanText,os.path.getsize(os.path.join(dir_path, audioPartFilename))]) 
                                speaker_id = os.path.basename(dir_path) + "_" + turnSpeaker + "_id_" + str(speakers_id[turnSpeaker])
                                # print("Speaker id written will be ", str(speaker_id), "for", dir_path)
                                outputList.append([os.path.join(dir_path, audioPartFilename),cleanText,os.path.getsize(os.path.join(dir_path, audioPartFilename)),speaker_id]) 
                        audioPartId+= 1
            except ValueError as error:
                print(error)
            except:
                e = sys.exc_info()[0]
                print(e)
                print("--------------------- Error in trs. One turn has been passed")

    regex = re.match(r"[a-zA-Z]*([0-9])*_.*",audioFilenameWe)
    splitFilename = audioFilenameWe.split(regex.group(1))
    print("This is the split :", splitFilename)
    if (len(splitFilename) == 2):
        previous_speakers = {}
        previous_speakers['prefix'] = splitFilename[0]
        previous_speakers['suffix'] = splitFilename[1]
        previous_speakers['speakers'] = speakers_id
    return outputList, speaker_index, previous_speakers


def check_file_length(wav_filename,label):    
    frames = 0
    frames = int(subprocess.check_output(['sox', '--i','-s', wav_filename], stderr=subprocess.STDOUT))
    if int(frames/SAMPLE_RATE*1000/10/2) < len(str(label)):
        return False
    elif frames/SAMPLE_RATE > MAX_SECS:
        return False
    else:
        return True

def clean_label(label):
    label = label.strip()
    label = label.lower()
    label = unicodedata.normalize('NFKC', label)
    label = re.sub(r"‘(.+)’", '', label, 0, re.MULTILINE)
    label = label.replace("ß", "b")
    label = label.replace("ă", "a")    
    label = label.replace("á", "a")
    label = label.replace("â", "a")
    label = label.replace("ã", "a")
    label = label.replace("ã", "a")    
    label = label.replace("å", "a")
    label = label.replace("ā", "a")
    label = label.replace("ä", "a")
    label = label.replace("ă", "a")
    label = label.replace("ạ", "a")
    label = label.replace("ạ", "a")
    label = label.replace("ả", "a")
    label = label.replace("ắ", "a")
    label = label.replace("ậ", "a")
    label = label.replace("ầ", "a")
    
    
    label = label.replace("ą", "a")            
    label = label.replace("æ", "ae")        
    #label = label.replace("ç", "c")
    label = label.replace("č", "c")
    label = label.replace("ҫ", "c")
    label = label.replace("ć", "c")
    label = label.replace("ċ", "c")
    
    label = label.replace("ḍ", "d")
            
    #label = label.replace("è", "e")
    #label = label.replace("é", "e")
    #label = label.replace("ê", "e")
    label = label.replace("ë", "e")
    label = label.replace("ě", "e")
    label = label.replace("ễ", "e")
    label = label.replace("ę", "e")
    label = label.replace("E", "e")
    label = label.replace("E", "e")
    label = label.replace("ē", "e")
    label = label.replace("ệ", "ê")                    
    label = label.replace("ġ", "g")
    label = label.replace("ğ", "g")
    label = label.replace("ħ", "h")
    
    label = label.replace("í", "i")
    label = label.replace("ì", "i")
    label = label.replace("ị", "i")
    label = label.replace("ľ", "i")
    
    
    
    #label = label.replace("î", "i")
    #label = label.replace("ï", "i")
    label = label.replace("ī", "i")
    label = label.replace("κ", "k")
    label = label.replace("к", "k")    
    label = label.replace("ļ", "l")
    label = label.replace("ľ ", "l")
    label = label.replace("ጠ", "m")
    label = label.replace("ñ", "n")
    label = label.replace("ǹ", "n")
    label = label.replace("ņ", "n")
    label = label.replace("ň", "n")
    label = label.replace("N", "n")
    label = label.replace("ṇ", "n")
    label = label.replace("א", "n")
    label = label.replace("ṅ", "n")
    
    
            
    label = label.replace("ò", "o")
    label = label.replace("ó", "o")
    #label = label.replace("ô", "o")
    label = label.replace("ö", "o")
    label = label.replace("ø", "o")
    label = label.replace("ō", "o")
    label = label.replace("õ", "o")
    label = label.replace("ợ", "o")
    label = label.replace("δ", "o")
    label = label.replace("ộ", "o")
    label = label.replace("☉", "o")
    label = label.replace("ŏ", "o")
    label = label.replace("ő", "o")
    label = label.replace("ð", "o")
    label = label.replace("ổ", "o")
    label = label.replace("ǫ", "o")
    label = label.replace("ồ", "o")
    label = label.replace("ơ", "o")
    
    
    
    
    label = label.replace("þ", "p")
    
    
    
    label = label.replace("π", "pi")       
    label = label.replace("ω", "omega")
    label = label.replace("ζ", "zeta")
        
    label = label.replace("ù", "u")
    label = label.replace("ử", "u")
    label = label.replace("υ", "u")
    label = label.replace("ū", "u")
    label = label.replace("ʉ", "u")
    label = label.replace("û", "u")
    label = label.replace("ú", "u")
    label = label.replace("ư", "u")
    
    
                 
    label = label.replace("đ", "d")
    label = label.replace("ደ", "da")
    label = label.replace("ı", "")
    label = label.replace("ł", "")
    label = label.replace("ń", "n")
    label = label.replace("ν", "n")    
    label = label.replace("œ", "oe")
    label = label.replace("ř", "r")
    label = label.replace("R", "r")    
    label = label.replace("г", "r")    
    label = label.replace("š", "s")
    label = label.replace("ṣ", "s")
    label = label.replace("ś", "s")
    label = label.replace("ş", "s")
    label = label.replace("ș", "s")
        
    label = label.replace("ṯ", "t")
    label = label.replace("ṭ", "t")
    label = label.replace("ť", "t")
    label = label.replace("τ", "t")
    
    
    label = label.replace("ț", "t")                
    label = label.replace("ż", "z")
    label = label.replace("ž", "z")
    label = label.replace("ź", "z")
    label = label.replace("α", "a")
    label = label.replace("å", "a")
    
    label = label.replace("β", "b")
    label = label.replace("γ", "y")
    label = label.replace("ý", "y")
    label = label.replace("ỳ", "y")
    
    
    label = label.replace("μ", "u")
    label = label.replace("ų", "u")
    label = label.replace("Z", "z")
    
        
    label = label.replace("–", "")
    label = label.replace("„", "")
    
    label = label.replace("—", "")
    label = label.replace("±", "")
    label = label.replace("·", "")        
    label = label.replace("’", "'")
    label = label.replace("′", "'")
    label = label.replace("`", "'")
    label = label.replace("ʼ", "'")
    label = label.replace("“", "")
    label = label.replace("”", "")
    label = label.replace("…", "")
    label = label.replace("-", " ")
    label = label.replace("_", " ")
    label = label.replace("†", " ")
    
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace("‘", "'")      
    label = label.replace(";", "")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace("ǃ", "")    
    label = label.replace(":", "")
    label = label.replace("\"", "")
    label = label.replace("!", "")
    label = label.replace("|", "")    
    label = label.replace("(", "")
    label = label.replace(")", "")
    label = label.replace("{", "")
    label = label.replace("}", "")
    label = label.replace("+", "")
    label = label.replace("/", "")
    label = label.replace(":", "")
    label = label.replace(";", "")
    label = label.replace("=", "")
    label = label.replace("ː", "")
    label = label.replace("£", "")
    label = label.replace("«", "")
    label = label.replace("°", "")
    label = label.replace("º", "")
    label = label.replace("»", "")
    label = label.replace("½", "")
    label = label.replace("ʿ", "")
    label = label.replace("ʿ", "")
    label = label.replace("ʾ", "")
    label = label.replace("ʻ", "")
    label = label.replace("‹", "")    
    label = label.replace("›", "")
    label = label.replace("я", "r à l'envers")
    
    
       
    label = label.replace("$", "dollar")
    label = label.replace("€", "euro")
    label = label.replace("1⁄2", "demi")    
    label = label.replace("⁄", " ")
    label = label.replace("1", "un")
    label = label.replace("2", "deux")
    label = label.replace("3", "trois")
    label = label.replace("4", "quatre")
    label = label.replace("5", "cinq")
    label = label.replace("%programfiles%", "programfiles")
    label = label.replace("%", "pourcent")
            

    return label      
    
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Import tcof corpus')
    PARSER.add_argument('corpus_dir', help='Directory contening corpus')
    PARSER.add_argument('--r',action='store_true', help='relative path')

    PARAMS = PARSER.parse_args()

    CORPUS_DIR = PARAMS.corpus_dir
    
    CORPUS_DIR = path.abspath(CORPUS_DIR)
    
    outputList = []
    print("CORPUS_DIR : ", CORPUS_DIR)
    outputList, speaker_index, previous_speakers = process_directory(CORPUS_DIR)
    export_csv(outputList,CORPUS_DIR)





    