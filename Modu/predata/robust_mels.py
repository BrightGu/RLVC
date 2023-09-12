import math
import os
import random
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pyrubberband as pyrb
import random
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import pickle
import yaml

import numpy as np

def load_wav(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(str(full_path))
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
    # return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output



def mel_normalize(S,clip_val=1e-5):
    S = (S - torch.log(torch.Tensor([clip_val])))*1.0/(0-torch.log(torch.Tensor([clip_val])))
    return S
#
#
def mel_denormalize(S,clip_val=1e-5):
    S = S*(0-torch.log(torch.Tensor([clip_val])).cuda()) + torch.log(torch.Tensor([clip_val])).cuda()
    return S



mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # print('min value is ', torch.min(y))
    # print('max value is ', torch.max(y))
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)


    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)


    return spec


def get_dataset_filelist(wavs_dir):
    figure_wavs={}
    for figure in os.listdir(wavs_dir):
        figure_dir = os.path.join(wavs_dir,figure)
        figure_wavs[figure] = [os.path.join(figure_dir, file_name) for file_name in os.listdir(figure_dir)]
    return figure_wavs

def get_test_dataset_filelist(wavs_dir):
    file_list = [os.path.join(wavs_dir, file_name) for file_name in os.listdir(wavs_dir)]
    choice_list = []
    for i in range(50):
        source_file = random.choice(file_list)
        target_file = random.choice(file_list)
        choice_list.append([source_file,target_file])
    return choice_list




out_dir = r"./datasets/Data/"
# train_wav_dir = r"/data/gyw_data/workspace/audio_datasets/VCTK-Corpus/WAV22050/wav22050/"
train_wav_dir = r"./vctk20_44100/"

def get_hifi_mels(config):
    figure_wavs_map = get_dataset_filelist(train_wav_dir)
    # spkemb_map= {}
    mel_label_map = {}
    for figure, file_list in figure_wavs_map.items():
        mel_label_list = []
        print("figure:", figure)
        for filename in file_list:
            try:
                file_label = os.path.basename(filename).split(".")[0]
                audio, sampling_rate = load_wav(filename)
                #### remove slice
                clip_audio, _ = librosa.effects.trim(audio, top_db=20)
                clip_audio = normalize(clip_audio) * 0.95
                clip_audio = torch.FloatTensor(clip_audio).unsqueeze(0)
                clip_mel = mel_spectrogram(clip_audio, config["n_fft"], config["num_mels"], config["sampling_rate"],
                                      config["hop_size"], config["win_size"], config["fmin"], config["fmax"],
                                      center=False)
                clip_mel = clip_mel.squeeze(0).transpose(0, 1)
                clip_mel = mel_normalize(clip_mel)
                mel_label_list.append([file_label,clip_mel])  # p255_106, file_label
            except:
                print("filename:", filename)
        mel_label_map[figure] = mel_label_list

    with open(os.path.join(out_dir, 'figure_label_mel_map29.pkl'), 'wb') as handle:
        pickle.dump(mel_label_map, handle)

if __name__ == '__main__':
    config_path = r"./hifi_config.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    get_hifi_mels(config)







