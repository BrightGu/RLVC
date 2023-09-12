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

def get_dataset_filelist(wavs_dir):
    file_list = []
    for figure in os.listdir(wavs_dir):
        figure_dir = os.path.join(wavs_dir, figure)
        for file_name in os.listdir(figure_dir):
            file_list.append(os.path.join(figure_dir, file_name))
    return file_list

def get_result_dataset_filelist(wavs_dir):
    choice_list = []
    work_dir = os.path.join(wavs_dir, "vctk20")
    figure_list = os.listdir(work_dir)
    for i in range(1500):
        src_fig = random.choice(figure_list)
        tar_fig = random.choice(figure_list)
        src_fig_dir = os.path.join(work_dir,src_fig)
        src_file_name = random.choice(os.listdir(src_fig_dir))
        tar_fig_dir = os.path.join(work_dir, tar_fig)
        tar_file_name = random.choice(os.listdir(tar_fig_dir))
        src_file_path = os.path.join(src_fig_dir, src_file_name)
        tar_file_path = os.path.join(tar_fig_dir, tar_file_name)
        label = src_fig[0]+"2"+tar_fig[0]+"_"+src_file_name.split(".")[0]+"TO"+tar_file_name.split(".")[0]
        choice_list.append([src_file_path, tar_file_path,label])

    work_dir = os.path.join(wavs_dir, "vcc20")
    figure_list = os.listdir(work_dir)
    for i in range(1500):
        src_fig = random.choice(figure_list)
        tar_fig = random.choice(figure_list)
        src_fig_dir = os.path.join(work_dir, src_fig)
        src_file_name = random.choice(os.listdir(src_fig_dir))
        tar_fig_dir = os.path.join(work_dir, tar_fig)
        tar_file_name = random.choice(os.listdir(tar_fig_dir))
        src_file_path = os.path.join(src_fig_dir, src_file_name)
        tar_file_path = os.path.join(tar_fig_dir, tar_file_name)
        label = src_fig[2] + "2" + tar_fig[2] + "_" + src_file_name.split(".")[0] + "TO" + tar_file_name.split(".")[0]
        choice_list.append([src_file_path, tar_file_path, label])

    work_dir = os.path.join(wavs_dir, "librispeech20")
    figure_list = os.listdir(work_dir)
    for i in range(1500):
        src_fig = random.choice(figure_list)
        tar_fig = random.choice(figure_list)
        src_fig_dir = os.path.join(work_dir, src_fig)
        src_file_name = random.choice(os.listdir(src_fig_dir))
        tar_fig_dir = os.path.join(work_dir, tar_fig)
        tar_file_name = random.choice(os.listdir(tar_fig_dir))
        src_file_path = os.path.join(src_fig_dir, src_file_name)
        tar_file_path = os.path.join(tar_fig_dir, tar_file_name)
        label = src_fig[0] + "2" + tar_fig[0] + "_" + src_file_name.split(".")[0] + "TO" + tar_file_name.split(".")[0]
        choice_list.append([src_file_path, tar_file_path, label])

    work_dir = os.path.join(wavs_dir, "aishell20")
    figure_list = os.listdir(work_dir)
    for i in range(1500):
        src_fig = random.choice(figure_list)
        tar_fig = random.choice(figure_list)
        src_fig_dir = os.path.join(work_dir, src_fig)
        src_file_name = random.choice(os.listdir(src_fig_dir))
        tar_fig_dir = os.path.join(work_dir, tar_fig)
        tar_file_name = random.choice(os.listdir(tar_fig_dir))
        src_file_path = os.path.join(src_fig_dir, src_file_name)
        tar_file_path = os.path.join(tar_fig_dir, tar_file_name)
        label = src_fig[0] + "2" + tar_fig[0] + "_" + src_file_name.split(".")[0] + "TO" + tar_file_name.split(".")[0]
        choice_list.append([src_file_path, tar_file_path, label])

    return choice_list

def get_cross_result_dataset_filelist(wavs_dir):
    choice_list = []

    src_cn_dir = os.path.join(wavs_dir, "aishell20")
    cn_figure_list = os.listdir(src_cn_dir)
    tar_en_dir = os.path.join(wavs_dir, "vctk20")
    en_figure_list = os.listdir(tar_en_dir)
    for i in range(2000):
        src_fig = random.choice(cn_figure_list)
        tar_fig = random.choice(en_figure_list)


        src_fig_dir = os.path.join(src_cn_dir,src_fig)
        src_file_name = random.choice(os.listdir(src_fig_dir))

        tar_fig_dir = os.path.join(tar_en_dir, tar_fig)
        tar_file_name = random.choice(os.listdir(tar_fig_dir))

        src_file_path = os.path.join(src_fig_dir, src_file_name)
        tar_file_path = os.path.join(tar_fig_dir, tar_file_name)
        label = src_fig[0]+"2"+tar_fig[0]+"_"+src_file_name.split(".")[0]+"TO"+tar_file_name.split(".")[0]
        choice_list.append([src_file_path, tar_file_path,label])

    return choice_list


def get_visual_test_filelist(wavs_dir):
    src_dir = "/home/gyw/workspace/VC_GAN/UnetVC/RLVC_Visual_Sample/source30/"
    tar_dir = "/home/gyw/workspace/VC_GAN/UnetVC/RLVC_Visual_Sample/target18/"
    src_list = [os.path.join(src_dir, item) for item in os.listdir(src_dir)]
    tar_list = [os.path.join(tar_dir, item) for item in os.listdir(tar_dir)]
    choice_list = []
    for src_file in src_list:
        for tar_file in tar_list:
            choice_list.append([src_file, tar_file])
    return choice_list

def get_test_dataset_filelist2(wavs_dir):
    file_list = [os.path.join(wavs_dir, file_name) for file_name in os.listdir(wavs_dir)]
    choice_list = []
    for i in range(60):
        source_file = random.choice(file_list)
        target_file = random.choice(file_list)
        choice_list.append([source_file, target_file])
    return choice_list


def get_infer_test_dataset_filelist(wavs_dir):
    src_path_list = []
    tar_path_list = []
    for file_name in os.listdir(wavs_dir):
        if 'src' in file_name:
            src_path_list.append(os.path.join(wavs_dir, file_name))
        else:
            tar_path_list.append(os.path.join(wavs_dir, file_name))
    src_path_list.sort()
    tar_path_list.sort()
    assert len(src_path_list) == len(tar_path_list)
    choice_list = []
    for i in range(len(src_path_list)):
        choice_list.append([src_path_list[i], tar_path_list[i]])
    return choice_list


def get_infer_test_dataset_filelist(wavs_dir):
    figure_list = os.listdir(wavs_dir)
    choice_list = []
    for i in range(1000):
        source_figure = random.choice(figure_list)
        source_figure_dir = os.path.join(wavs_dir, source_figure)
        source_file_list = [os.path.join(source_figure_dir, file_name) for file_name in os.listdir(source_figure_dir)]
        source_file = random.choice(source_file_list)

        target_figure = random.choice(figure_list)
        while target_figure==source_figure:
            target_figure = random.choice(figure_list)
        target_figure_dir = os.path.join(wavs_dir, target_figure)
        target_file_list = [os.path.join(target_figure_dir, file_name) for file_name in os.listdir(target_figure_dir)]
        target_file = random.choice(target_file_list)

        choice_list.append([source_file, target_file])

    return choice_list

def get_infer_test_dataset_filelist2(wavs_dir):
    file_list = os.listdir(wavs_dir)
    choice_list = []
    for i in range(1000):
        source_name = random.choice(file_list)
        source_file = os.path.join(wavs_dir, source_name)
        target_name = random.choice(file_list)
        target_file = os.path.join(wavs_dir, target_name)

        choice_list.append([source_file, target_file])
    return choice_list

def get_infer_test_dataset_filelist3(src_wavs_dir,tar_wavs_dir):
    src_file_list = os.listdir(src_wavs_dir)
    tar_file_list = os.listdir(tar_wavs_dir)

    r = random.random
    random.seed(56)
    random.shuffle(tar_file_list, random=r)

    choice_list = []
    for index in range(min(len(src_file_list),len(tar_file_list))):
        source_file = os.path.join(src_wavs_dir, src_file_list[index])
        target_file = os.path.join(tar_wavs_dir, tar_file_list[index])
        choice_list.append([source_file, target_file])
    return choice_list

def get_visual_test_dataset_filelist(wavs_dir):
    source_dir = os.path.join(wavs_dir, 'source')
    target_dir = os.path.join(wavs_dir, 'target')
    source_file_list = os.listdir(source_dir)
    target_file_list = os.listdir(target_dir)

    choice_list = []
    for source_file in source_file_list:
        source_file_path = os.path.join(source_dir, source_file)
        target_file = random.choice(target_file_list)
        target_file_path = os.path.join(target_dir, target_file)
        choice_list.append([source_file_path, target_file_path])
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
