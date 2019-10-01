import os
import collections
import numpy as np
import torch.utils.data

from utils.text import text_to_sequence, phoneme_to_sequence, pad_with_eos_bos


class TTSDataset(torch.utils.data.Dataset):

    def __init__(self, outputs_per_step, text_cleaner, ap, meta_data,
                 use_phonemes=True, phoneme_cache_path=None,
                 phoneme_language='en-us'):
        self.outputs_per_step = outputs_per_step
        self.cleaners = text_cleaner
        self.ap = ap
        self.items = meta_data

        self.use_phonemes = use_phonemes
        self.phoneme_cache_path = phoneme_cache_path
        self.phoneme_language = phoneme_language

        if use_phonemes and not os.path.isdir(phoneme_cache_path):
            os.makedirs(phoneme_cache_path, exist_ok=True)

        print('DataLoader initialization')
        print('  Use phonemes: {}'.format(self.use_phonemes))
        print('  Number of instances: {}'.format(len(self.items)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def load_data(self, idx):
        text, wav_file = self.items[idx]
        wav = np.asarray(self.ap.load_wav(wav_file), dtype=np.float32)

        if self.use_phonemes:
            text = self._load_or_generate_phoneme_sequence(wav_file, text)
        else:
            text = np.asarray(
                text_to_sequence(text, [self.cleaners]), dtype=np.int32)

        assert text.size > 0, self.items[idx][1]
        assert wav.size > 0, self.items[idx][1]

        sample = {
            'text': text,
            'wav': wav,
            'item_idx': self.items[idx][1]
        }

        return sample

    def _load_or_generate_phoneme_sequence(self, wav_file, text):
        filename = os.path.basename(wav_file).split('.')[0]
        cache_path = os.path.join(
            self.phoneme_cache_path, filename + '_phoneme.npy')

        try:
            phonemes = np.load(cache_path)
        except FileNotFoundError:
            phonemes = self._generate_and_cache_phoneme_sequence(
                text, cache_path)

        return phonemes

    def _generate_and_cache_phoneme_sequence(self, text, cache_path):
        phonemes = phoneme_to_sequence(text, [self.cleaners],
                                       language=self.phoneme_language,
                                       enable_eos_bos=False)
        phonemes = np.asarray(phonemes, dtype=np.int32)
        np.save(cache_path, phonemes)
        return phonemes

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
        text_lengths = np.array([len(d['text']) for d in batch])
        text_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor(text_lengths), dim=0, descending=True)

        # テキストに合わせて他もソート
        wav = [batch[idx]['wav'] for idx in ids_sorted_decreasing]
        item_idxs = [batch[idx]['item_idx'] for idx in ids_sorted_decreasing]
        text = [batch[idx]['text'] for idx in ids_sorted_decreasing]

        # 音声をメルスペクトログラムに変換
        mel = [self.ap.melspectrogram(w).astype('float32') for w in wav]

        # 音声を線形スペクトログラムに変換
        linear = [self.ap.spectrogram(w).astype('float32') for w in wav]

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
            stop_targets, item_idxs


def preprocess_ljspeech(root_path, meta_file):
    text_file = os.path.join(root_path, meta_file)
    items = []
    with open(text_file, 'r') as fp:
        for line in fp:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file])
    return items


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
