# https://github.com/HLTSingapore/Emotional-Speech-Data
import os
from pypinyin import lazy_pinyin, Style
import text
from pypinyin.style._utils import get_initials, get_finals
from tqdm import tqdm
import shutil

import textgrid
import pandas as pd
# import librosa
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import read


class ESD:
    def __init__(self, lang):
        assert lang == "zh" or lang == "en"
        self.lang = lang  # language
        self.encoding = ['gbk', 'gbk', 'utf-16-le', 'gbk', 'gbk', 'utf-16-le', 'utf-16-le', 'utf-16-le', 'gbk',
                         'utf-16-le'] + \
                        ['utf-8', 'utf-16-le', 'utf-16-le', 'utf-16-le', 'utf-8', 'gbk', 'gbk', 'utf-16-le',
                         'utf-16-le', 'utf-8']
        self.trans = {'中立': 'Neutral', '生气': 'Angry', '快乐': 'Happy', '伤心': 'Sad', '惊喜': 'Surprise'}
        self.division = ['train', 'test', 'evaluation']

    def create_filelists_from_mfa(self, sid_start=0):
        def get_emotion(label):
            i = (int(label) - 1) // 350
            if i == 0:
                return "Neutral"
            elif i == 1:
                return "Angry"
            elif i == 2:
                return "Happy"
            elif i == 3:
                return "Sad"
            elif i == 4:
                return "Surprise"
            else:
                raise ValueError("Invalid filename: %s" % str_)

        # prepare for VITS model
        # path | speaker | text | emotion
        division = ['train', 'test', 'evaluation']
        audiopaths_sid_text_emotion = {}
        for div in division:
            audiopaths_sid_text_emotion[div] = []

        # 1. read wavfile paths & divide the dataset
        for speaker in range(10):
            speaker_path = os.path.join("/home/weili/data/ESD_", str(speaker))
            for audio in os.listdir(speaker_path):
                if audio[-4:] == ".wav":
                    audio_path = os.path.join(speaker_path, audio)  #
                    label = audio.split('_')[-1].split('.')[0]
                    ###
                    text_path = audio_path.replace("ESD_", "ESD").replace(".wav", ".lab")
                    with open(text_path, "r") as rf:
                        text = rf.readline()
                    emotion = get_emotion(label)
                    ###
                    if (int(label) - 1) % 350 < 20:
                        audiopaths_sid_text_emotion['evaluation'].append(
                            "|".join([audio_path, str(speaker + sid_start), text, emotion]))
                    elif (int(label) - 1) % 350 < 50:
                        audiopaths_sid_text_emotion['test'].append(
                            "|".join([audio_path, str(speaker + sid_start), text, emotion]))
                    else:
                        audiopaths_sid_text_emotion['train'].append(
                            "|".join([audio_path, str(speaker + sid_start), text, emotion]))

        # print(audiopaths_sid_text_emotion)
        # for div in division:
        #     print(len(audiopaths_sid_text_emotion[div]))
        # 5882, 568, 438

        # 2. create the filelists
        for div in division:
            with open("esd_audio_sid_text_emotion_%s_filelists.txt" % div, "w") as wf:
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

    def convert(self, keep_sr):
        for speaker in range(10):
            speaker_path = os.path.join("/home/weili/data/ESD", str(speaker))
            for audio in os.listdir(speaker_path):
                if audio[-4:] == ".wav":
                    audio_path = os.path.join(speaker_path, audio)
                    # y, sr = librosa.load(audio_path)
                    sr, y = read(audio_path)
                    textgrid_path = audio_path.replace("/home/weili/data/ESD", "/home/weili/data/ESD_").replace(".wav",
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

    def find(self, path, sid, emotion, audio):
        # try to find the path of .wav file
        for div in self.division:
            speaker_path = os.path.join(path, '00%02d' % sid)
            audio_path = os.path.join(speaker_path, emotion, div, audio + '.wav')
            if os.path.exists(audio_path):
                return audio_path
        raise ValueError('File not found: %s' % audio_path)

    def iterator(self, path):
        # scan the dataset
        speakers = range(1, 11) if self.lang == "zh" else range(11, 21)
        for sid in tqdm(speakers):
            speaker_path = os.path.join(path, '00%02d' % sid)
            textfile = os.path.join(speaker_path, '00%02d' % sid + '.txt')
            with open(textfile, 'r', encoding=self.encoding[sid - 1]) as f:
                rf = f.readlines()
            for line in rf:
                # 1.1 scan textfile
                split = line.strip().split('\t')  # for english
                if len(split) != 3:  # remove blank line
                    continue
                audio, txt, emotion = split
                if len(audio) == 12:  # remove unknown blank in the head
                    audio = audio[1:]
                if self.lang == 'zh':
                    emotion = self.trans[emotion]  # zh2en
                yield audio, sid, txt, emotion

    def prepare_mfa(self, path="/home/weili/data/Emotional Speech Dataset (ESD)", target_path="/home/weili/data/ESD"):
        for i in range(10):
            os.makedirs(os.path.join(target_path, str(i)), exist_ok=True)

        for audio, sid, text, emotion in self.iterator(path):
            audio_path = self.find(path, sid, emotion, audio)
            audio_path_target = os.path.join(target_path, str(sid - 1 if self.lang == "zh" else sid - 11),
                                             audio_path.split("/")[-1])
            text_path_target = audio_path_target.replace(".wav", ".lab")
            # print(f"copy from {audio_path} -> {audio_path_target}")
            # print(text)
            with open(text_path_target, "w") as wf:
                wf.write(text)
            shutil.copy(audio_path, audio_path_target)

    def create_filelists(self, path, cleaners=None):
        # 1. define variables
        if self.lang == "zh":
            _initials = []
            _finals = []
            _tones = []
            _punctuation_zh = []
        audiopaths_sid_text_emotion = {}
        for div in self.division:
            audiopaths_sid_text_emotion[div] = []

        # 2. create filelists
        for audio, sid, text, emotion in self.iterator(path):
            # 2.1 clean the texts
            if cleaners:  # clean the txt into phonemes
                cleaned = text._clean_text(txt, [cleaners])  # pass a list of cleaners
                if self.lang == 'zh':  # write dictionary step-1
                    for item in cleaned:
                        if item[-1].isdigit():
                            if item[:-1] not in _finals:
                                _finals.append(item[:-1])
                            if item[-1] not in _tones:
                                _tones.append(item[-1])
                        elif not item[0].isalnum():
                            for p in item:
                                if p not in _punctuation_zh:
                                    _punctuation_zh.append(p)
                        else:
                            if get_initials(item, strict=True) != '':
                                if item not in _initials:
                                    _initials.append(item)
                            else:  # qing sheng
                                if item not in _finals:
                                    _finals.append(item)
                    txt = ' '.join(cleaned)
                elif self.lang == 'en':
                    txt = cleaned

            # 2.2 save
            audiopaths_sid_text_emotion[div].append(
                [self.find(path, sid, emotion, audio), str(sid - 1 if self.lang == "zh" else sid - 11), txt, emotion])

        # 3. write filelists
        extension = 'cleaned' if cleaners else 'txt'
        for div in self.division:  # write filelists
            with open('filelists/raw/esd_audio_sid_text_emo_%s_%s_filelist.%s' % (div, self.lang, extension), 'w',
                      encoding='utf-8') as f:
                f.writelines(['|'.join(x) + '\n' for x in audiopaths_sid_text_emotion[div]])

        # [4]. create pronunciation dictionary for Chinese dataset
        if self.lang == 'zh':
            with open('text/meta.py', 'w', encoding='utf-8') as f:  # write dictionary step-2
                _initials.sort()
                _finals.sort()
                _tones.sort()
                _punctuation_zh.sort()
                f.writelines("_initials = [\"%s\"]" % '", "'.join(_initials) + '\n')
                f.writelines("_finals = [\"%s\"]" % '", "'.join(_finals) + '\n')
                f.writelines("_tones = [\"%s\", \"\"]" % '", "'.join(_tones) + '\n')
                f.writelines("_punctuation_zh = [\"%s\"]" % '", "'.join(_punctuation_zh) + '\n')


if __name__ == '__main__':
    dataset = ESD(lang="en")
    # dataset.create_filelists("/home/weili/data/Emotional Speech Dataset (ESD)")
    # dataset.prepare_mfa()
    dataset.convert(keep_sr=False)
    # dataset.create_filelists_from_mfa()

    # mfa validate /home/weili/data/ESD english_us_arpa english_us_arpa

    # mfa g2p /home/weili/Documents/MFA/ESD/oovs_found_english_us_arpa.txt english_us_arpa /home/weili/data/ESD_/g2pped_oovs.txt --dictionary_path english_us_arpa

    # mfa model add_words english_us_arpa /home/weili/data/ESD_/g2pped_oovs.txt

    # mfa validate /home/weili/data/ESD english_us_arpa english_us_arpa --clean

# mfa align /home/weili/data/ESD english_us_arpa english_us_arpa /home/weili/data/ESD_

# nohup python process_esd.py > log_proc.txt 2>&1 &


# 1.0 <emotion id>, GST
# 1.1 <strengthnet: ranking function>
# 1.2 <SER-based emotion intensity predictor>

# 1.3 arousal valence: difficult to prepare annotations
# acoustic morpheme?

# 2.0 voice conversion of vits is natural with reference encoder (audio, eid as input) <not natural in fact?>
# 2.1 separate prosody and emotion?

# 3.0 syntax information to control the duration of the speech (totally different field)

