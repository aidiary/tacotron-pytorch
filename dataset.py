import os
import collections
import numpy as np
import torch.utils.data

from utils.text import text_to_sequence, phoneme_to_sequence, pad_with_eos_bos


class TTSDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, target_list, outputs_per_step):
        self.data_root = data_root
        self.target_list = target_list
        self.outputs_per_step = outputs_per_step

        self.phoneme_dir = os.path.join(self.data_root, 'phoneme')
        self.melspec_dir = os.path.join(self.data_root, 'melspec')
        self.spec_dir = os.path.join(self.data_root, 'spec')

        print('DataLoader initialization')
        print('  data_root: {}'.format(self.data_root))
        print('  target_list: {}'.format(self.target_list))

        self.targets = []
        with open(os.path.join(data_root, target_list), 'r') as fp:
            for line in fp:
                line = line.rstrip()
                self.targets.append(line)
        print(len(self.targets))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        prefix = self.targets[idx]
        return self.load_data(prefix)

    def load_data(self, prefix):
        phoneme = np.load(os.path.join(
            self.data_root, 'phoneme', prefix + '.npy'))
        spec = np.load(os.path.join(self.data_root, 'spec', prefix + '.npy'))
        melspec = np.load(os.path.join(
            self.data_root, 'melspec', prefix + '.npy'))
        sample = {
            'prefix': prefix,
            'phoneme': phoneme,
            'melspec': melspec,
            'spec': spec
        }
        return sample

    def collate_fn(self, batch):
        """
            Perform preprocessing and create a final data batch:
            1. PAD sequences with the longest sequence in the batch
            2. Convert Audio signal to Spectrograms.
            3. PAD sequences that can be divided by r.
            4. Convert Numpy to Torch tensors.
        """
        assert isinstance(batch[0], collections.Mapping)

        # テキスト長の長い順にサンプルをソート
        text_lengths = np.array([len(d['phoneme']) for d in batch])
        text_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor(text_lengths), dim=0, descending=True)

        # テキストに合わせて他もソート
        text = [batch[idx]['phoneme'] for idx in ids_sorted_decreasing]
        prefix = [batch[idx]['prefix'] for idx in ids_sorted_decreasing]
        mel = [batch[idx]['melspec'] for idx in ids_sorted_decreasing]
        linear = [batch[idx]['spec'] for idx in ids_sorted_decreasing]

        # paddingする前のmelの長さを返す？
        mel_lengths = [m.shape[1] + 1 for m in mel]  # +1 for zero-frame

        # stop tokenターゲットを計算
        # stopしないフレームには0を格納
        stop_targets = [
            np.array([0.0] * (mel_len - 1)) for mel_len in mel_lengths
        ]

        # stop targetsをpadding
        stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

        # batchの最長系列に合わせてpadding
        text = prepare_data(text).astype(np.int32)

        # スペクトログラムをpadding (batch, input_size, seq_len)
        linear = prepare_tensor(linear, self.outputs_per_step)
        mel = prepare_tensor(mel, self.outputs_per_step)
        assert linear.shape[2] == mel.shape[2]
        timesteps = mel.shape[2]

        # (batch, seq_len, input_size)
        linear = linear.transpose(0, 2, 1)
        mel = mel.transpose(0, 2, 1)

        text_lengths = torch.LongTensor(text_lengths)
        text = torch.LongTensor(text)
        linear = torch.FloatTensor(linear).contiguous()
        mel = torch.FloatTensor(mel).contiguous()
        mel_lengths = torch.LongTensor(mel_lengths)
        stop_targets = torch.FloatTensor(stop_targets)

        return text, text_lengths, linear, mel, mel_lengths, \
            stop_targets, prefix


def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[1] for x in inputs)) + 1
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])


def prepare_stop_target(inputs, out_steps):
    max_len = max((x.shape[0] for x in inputs)) + 1
    # out_steps単位でフレームを出力するのでその数で割れるようにpaddingする
    # 停止するフレームでは1.0が格納される
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])


def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]),
                  mode='constant', constant_values=_pad)


def _pad_tensor(x, length):
    _pad = 0
    assert x.ndim == 2
    return np.pad(x, [[0, 0], [0, length - x.shape[1]]],
                  mode='constant', constant_values=_pad)


def _pad_stop_target(x, length):
    _pad = 1.0
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]),
                  mode='constant', constant_values=_pad)
