import os
import shutil
import requests
import tarfile

import textgrid
import pandas as pd
# import librosa
import numpy as np
from scipy.io import wavfile


class Emov:
    def __init__(self, download_path="EMOV-DB", output_path="EMOV"):
        self.download_path = download_path
        self.output_path = output_path

    def create_filelists(self, sid_start=10):
        # prepare for VITS model
        # path | speaker | text | emotion
        division = ['train', 'test', 'evaluation']
        audiopaths_sid_text_emotion = {}
        for div in division:
            audiopaths_sid_text_emotion[div] = []

        # 1. read raw sentences
        with open(self.download_path + "/cmuarctic.data", "r") as rf:
            lines = rf.readlines()

        label_to_transcript = {}

        for line in lines:
            line = line.split('"')
            sent = line[1]
            label = line[0].rstrip().split('_')[-1]
            if label[0] == "b":
                continue
            label = label[1:]
            label_to_transcript[label] = sent

        # 2. read wavfile paths & divide the dataset
        for speaker in range(1, 5):
            speaker_path = os.path.join(self.output_path, str(speaker))
            for audio in os.listdir(speaker_path):
                if audio[-4:] == ".wav":
                    audio_path = os.path.join(os.getcwd(), speaker_path, audio)
                    sid = speaker - 1
                    label = audio.split('_')[-1].split('.')[0]
                    text = label_to_transcript[label]
                    emotion = audio.split('_')[0]
                    if int(label) < 35:
                        audiopaths_sid_text_emotion['test'].append(
                            "|".join([audio_path, str(sid + sid_start), text, emotion]))
                    elif int(label) < 60:
                        audiopaths_sid_text_emotion['evaluation'].append(
                            "|".join([audio_path, str(sid + sid_start), text, emotion]))
                    else:
                        audiopaths_sid_text_emotion['train'].append(
                            "|".join([audio_path, str(sid + sid_start), text, emotion]))

        # print(audiopaths_sid_text_emotion)
        # for div in division:
        #     print(len(audiopaths_sid_text_emotion[div]))
        # 5882, 568, 438

        # 3. create the filelists
        for div in division:
            with open("emov_audio_sid_text_emotion_%s_filelists.txt" % div, "w") as wf:
                wf.writelines([x + "\n" for x in audiopaths_sid_text_emotion[div]])

    def get_all_phone_with_timings(self, f='/home/weili/data/EMOV/1/amused_1-15_0001.TextGrid'):
        """get all phonemes of a sentence located in tg[1], and filter silence and empty parts, then convert to DataFrame
        """
        tg = textgrid.TextGrid.fromFile(f)
        # get phones and drop "sp", "sil" and empty strings
        phones = [[el.minTime, el.maxTime, el.mark] for el in tg[1] if el.mark not in ['sil', 'sp', '', 'spn']]
        phones = pd.DataFrame(phones)
        phones.columns = ["start", "end", "phone"]
        return phones

    def convert(self, keep_sr=False):
        for speaker in range(1, 5):
            speaker_path = os.path.join(self.download_path, str(speaker))
            for audio in os.listdir(speaker_path):
                if audio[-4:] == ".wav":
                    audio_path = os.path.join(speaker_path, audio)
                    sr, y = wavfile.read(audio_path)
                    textgrid_path = audio_path.replace(self.download_path, self.output_path).replace(".wav",
                                                                                                     ".TextGrid")
                    if os.path.exists(textgrid_path):
                        p = self.get_all_phone_with_timings(f=textgrid_path)
                    else:
                        # wavfile and textfile mismatch
                        continue

                    speech_segs = np.array([])

                    for interval in p.values:
                        speech_seg = y[int(interval[0] * sr): int(interval[1] * sr)]
                        speech_segs = np.append(speech_segs, speech_seg)

                    if keep_sr:
                        wavfile.write(textgrid_path.replace(".TextGrid", ".wav"), sr, speech_segs)
                    else:
                        wavfile.write(textgrid_path.replace(".TextGrid", ".wav"), 16000, speech_segs)

    def prepare_mfa(self, clean=False):
        def remove_punct(string):
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for x in string.lower():
                if x in punctuations:
                    string = string.replace(x, " ")

            return string.lower()

        # create the textfile with the same name of wavfile

        # 1. read transcripts
        with open(self.download_path + "/cmuarctic.data", "r") as rf:
            lines = rf.readlines()

        label_to_transcript = {}

        for line in lines:
            line = line.split('"')
            sent = line[1]
            label = line[0].rstrip().split('_')[-1]
            if label[0] == "b":
                continue
            label = label[1:]
            sent = remove_punct(sent)  # remove punct
            sent = sent.replace("1908", "nineteen o eight")
            sent = sent.replace("18", "eighteen")
            sent = sent.replace("16", "sixteen")
            sent = sent.replace("nightglow", "night glow")
            sent = sent.replace("mr ", "mister ")
            sent = sent.replace("mrs ", "misters ")
            sent = sent.replace("  ", " ")
            label_to_transcript[label] = sent

        # 2. scan wavfiles and create textfiles
        for speaker in range(1, 5):
            speaker_path = os.path.join(self.download_path, str(speaker))
            # for emotion in os.listdir(speaker_path):
            #     emotion_path = os.path.join(speaker_path, emotion)
            for audio in os.listdir(speaker_path):
                if audio[-4:] == ".wav":
                    textfile = audio[:-4] + ".lab"
                    label = audio.split('_')[-1].split('.')[0]
                    transcript = label_to_transcript[label]
                    if clean:
                        os.remove(os.path.join(speaker_path, textfile))
                    else:
                        with open(os.path.join(speaker_path, textfile), 'w') as wf:
                            wf.write(transcript)

    def download(self):
        download_links = [
            "https://www.openslr.org/resources/115/bea_Amused.tar.gz",
            "https://www.openslr.org/resources/115/bea_Angry.tar.gz",
            "https://www.openslr.org/resources/115/bea_Disgusted.tar.gz",
            "https://www.openslr.org/resources/115/bea_Neutral.tar.gz",
            "https://www.openslr.org/resources/115/bea_Sleepy.tar.gz",

            "https://www.openslr.org/resources/115/jenie_Amused.tar.gz",
            "https://www.openslr.org/resources/115/jenie_Angry.tar.gz",
            "https://www.openslr.org/resources/115/jenie_Disgusted.tar.gz",
            "https://www.openslr.org/resources/115/jenie_Neutral.tar.gz",
            "https://www.openslr.org/resources/115/jenie_Sleepy.tar.gz",

            "https://www.openslr.org/resources/115/josh_Amused.tar.gz",
            "https://www.openslr.org/resources/115/josh_Neutral.tar.gz",
            "https://www.openslr.org/resources/115/josh_Sleepy.tar.gz",

            "https://www.openslr.org/resources/115/sam_Amused.tar.gz",
            "https://www.openslr.org/resources/115/sam_Angry.tar.gz",
            "https://www.openslr.org/resources/115/sam_Disgusted.tar.gz",
            "https://www.openslr.org/resources/115/sam_Neutral.tar.gz",
            "https://www.openslr.org/resources/115/sam_Sleepy.tar.gz",

            "http://www.festvox.org/cmu_arctic/cmuarctic.data"
        ]

        target_directories = [

            self.download_path + "/1",
            self.download_path + "/1",
            self.download_path + "/1",
            self.download_path + "/1",
            self.download_path + "/1",

            self.download_path + "/2",
            self.download_path + "/2",
            self.download_path + "/2",
            self.download_path + "/2",
            self.download_path + "/2",

            self.download_path + "/3",
            self.download_path + "/3",
            self.download_path + "/3",

            self.download_path + "/4",
            self.download_path + "/4",
            self.download_path + "/4",
            self.download_path + "/4",
            self.download_path + "/4",

            self.download_path
        ]

        for directory in target_directories:
            os.makedirs(directory, exist_ok=True)

        for link, target_directory in zip(download_links, target_directories):
            filename = os.path.basename(link)
            file_path = os.path.join(target_directory, filename)

            response = requests.get(link, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"download successed:{filename}")

                if filename[-5:] != ".data":
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(path=target_directory)
                    os.remove(file_path)
            else:
                print(f"download failed:{filename}")


dataset = Emov()
# dataset.download()
# dataset.prepare_mfa()

# mfa validate /home/weili/data/EMOV-DB english_us_arpa english_us_arpa

# mfa g2p /home/weili/Documents/MFA/EMOV-DB/oovs_found_english_us_arpa.txt english_us_arpa /home/weili/data/EMOV/g2pped_oovs.txt --dictionary_path english_us_arpa

# mfa model add_words english_us_arpa /home/weili/data/EMOV/g2pped_oovs.txt

# mfa validate /home/weili/data/EMOV-DB english_us_arpa english_us_arpa --clean

# mfa align /home/weili/data/EMOV-DB english_us_arpa english_us_arpa /home/weili/data/EMOV

# dataset.convert(keep_sr=False)
dataset.create_filelists()