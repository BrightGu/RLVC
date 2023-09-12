"""Dataset for speaker embedding."""
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from Modu.data.infinite_dataloader import InfiniteDataLoader,infinite_iterator
class GE2EDataset(Dataset):
    """Sample utterances from speakers."""

    def __init__(
        self,
        data_path,
        # speaker_infos: dict,
        n_utterances: int,
        seg_len: int,

    ):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        self.speaker_infos=[]
        self.n_utterances = n_utterances
        self.seg_len = seg_len

        with open(data_path,"rb") as f:
            self.speaker_infos = pickle.load(f)# map key:p234 value_list: item: [label,mel]

        self.label_mel_list = []
        self.figures = list(self.speaker_infos.keys())
        for figure in self.figures:
            self.label_mel_list.append(self.speaker_infos[figure])
        self.len = len(self.figures)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label_feature_all = self.speaker_infos[self.figures[index]]
        while(len(label_feature_all)<self.n_utterances):
            figure = random.sample(self.figures,1)[0]
            label_feature_all = self.speaker_infos[figure]

        label_feature_list = random.sample(label_feature_all, self.n_utterances)
        labels = [label_feat[0] for label_feat in label_feature_list]
        feats =  [fixed_length(label_feat[1],self.seg_len) for label_feat in label_feature_list]
        # has remove slice
        return feats,feats

def fixed_length(mel, segment_len=128):
    if mel.shape[0] < segment_len:
        len_pad = segment_len - mel.shape[0]
        mel = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
        assert mel.shape[0] == segment_len
    elif mel.shape[0] > segment_len:
        left = np.random.randint(mel.shape[0] - segment_len)
        mel = mel[left:left + segment_len, :]
    return mel

def pad_mul_segment(mel, segment_len=128):
    pad_len = mel.shape[0] % segment_len
    if pad_len == 0:
        return mel
    pad_len = segment_len - pad_len
    pad_mel = mel[:pad_len, :]
    mul_mel = torch.cat([mel, pad_mel], 0)
    assert mul_mel.shape[0] % segment_len == 0
    return mul_mel

def collate_batch(batch):
    """Collate a whole batch of utterances."""
    feats_flatten = []
    spk_feats_flatten = []
    for item in batch:
        for u in item[0]:
            feats_flatten.append(torch.FloatTensor(u))
        for k in item[1]:
            spk_feats_flatten.append(torch.FloatTensor(k))
    feats_flatten = pad_sequence(feats_flatten, batch_first=True, padding_value=0)
    spk_feats_flatten = pad_sequence(spk_feats_flatten, batch_first=True, padding_value=0)
    return feats_flatten, spk_feats_flatten

def get_data_loaders(data_dir,batch_size,n_utterances):
    dataset = GE2EDataset(data_dir, n_utterances=n_utterances, seg_len=128)
    train_loader = InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        collate_fn=collate_batch,
        drop_last=True,
    )
    train_iter = infinite_iterator(train_loader)
    return train_iter