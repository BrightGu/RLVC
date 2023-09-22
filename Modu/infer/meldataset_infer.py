import os
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.utils.data
import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import random
import torch
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

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_normalize(S, clip_val=1e-5):
    S = (S - torch.log(torch.Tensor([clip_val]))) * 1.0 / (0 - torch.log(torch.Tensor([clip_val])))
    return S

def mel_denormalize(S, clip_val=1e-5):
    S = S * (0 - torch.log(torch.Tensor([clip_val])).cuda()) + torch.log(torch.Tensor([clip_val])).cuda()
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
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def get_test_dataset_filelist(wavs_dir):
    file_list = [os.path.join(wavs_dir, file_name) for file_name in os.listdir(wavs_dir)]
    choice_list = []
    for i in range(50):
        source_file = random.choice(file_list)
        target_file = random.choice(file_list)
        choice_list.append([source_file, target_file])
    return choice_list

def fixed_length(mel, segment_len=128):
    if mel.shape[0] < segment_len:
        len_pad = segment_len - mel.shape[0]
        mel = F.pad(mel, (0, 0, 0, len_pad), 'constant')
        assert mel.shape[0] == segment_len
    elif mel.shape[0] > segment_len:
        left = np.random.randint(mel.shape[0] - segment_len)
        mel = mel[left:left + segment_len, :]
    return mel


def collate_batch(batch):
    """Collate a batch of data."""
    input_mels, word = zip(*batch)
    ## speak_input_dim
    speak_input_mels = [fixed_length(input_mel) for input_mel in input_mels]
    speak_input_mels = torch.stack(speak_input_mels)

    input_lens = [len(input_mel) for input_mel in input_mels]
    overlap_lens = input_lens
    input_mels = pad_sequence(input_mels, batch_first=True)  # (batch, max_src_len, wav2vec_dim)
    input_masks = [torch.arange(input_mels.size(1)) >= input_len for input_len in input_lens]
    input_masks = torch.stack(input_masks)  # (batch, max_src_len)
    return speak_input_mels, input_mels, input_masks, word, overlap_lens

def pad_mul_segment(mel,segment_len=128):
    pad_len = mel.shape[0]%segment_len
    if pad_len==0:
        return mel
    pad_len = segment_len-pad_len

    long_mel_list = []
    num = pad_len//mel.shape[0]+1
    for i in range(num):
        long_mel_list.append(mel)
    long_mel = torch.cat(long_mel_list,0)
    pad_mel = long_mel[:pad_len, :]
    mul_mel = torch.cat([mel,pad_mel],0)
    assert mul_mel.shape[0]%segment_len==0
    return mul_mel

def w2v_load_wav(audio_path, sample_rate: int, trim: bool = False) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:  # prevent empty slice
            wav = wav[start:end]
    return wav

class Test_MelDataset(torch.utils.data.Dataset):
    def __init__(self, test_files, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, device=None):
        self.audio_files = test_files
        self.sampling_rate = sampling_rate

        self.n_fft = n_fft
        # self.num_mels = num_mels
        self.src_num_mels = 80
        self.tar_num_mels = 80
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.device = device

    def __getitem__(self, index):
        src_filename, tar_filename,convert_label = self.audio_files[index]
        ####### len
        src_audio, src_sampling_rate = load_wav(src_filename)
        ### split
        src_audio = normalize(src_audio) * 0.95
        src_audio = torch.FloatTensor(src_audio)
        src_audio = src_audio.unsqueeze(0)
        src_mel = mel_spectrogram(src_audio, self.n_fft, self.src_num_mels, self.sampling_rate, self.hop_size,
                                  self.win_size, self.fmin, self.fmax, center=False)
        src_mel = src_mel.squeeze(0).transpose(0, 1)
        src_mel = mel_normalize(src_mel)
        fake_mel = src_mel
        ###### tar
        tar_tensor, sample_rate = load_wav(tar_filename)
        clip_audio, _ = librosa.effects.trim(tar_tensor, top_db=20)
        clip_audio = normalize(clip_audio) * 0.95
        clip_audio = torch.FloatTensor(clip_audio)
        clip_audio = clip_audio.unsqueeze(0)
        clip_mel = mel_spectrogram(clip_audio, self.n_fft, self.tar_num_mels, self.sampling_rate, self.hop_size,self.win_size, self.fmin, self.fmax, center=False)# 1,80,167
        clip_mel = clip_mel.squeeze(0).transpose(0, 1)

        clip_mel = mel_normalize(clip_mel)
        # clip_mel = pad_mul_segment(clip_mel,128)
        return fake_mel, clip_mel, convert_label


    def __len__(self):
        return len(self.audio_files)
