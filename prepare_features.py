import os
import argparse
import numpy as np
from tqdm import tqdm
from utils.audio import AudioProcessor
from utils.text import phoneme_to_sequence


def load_metadata(metadata_file):
    items = []
    with open(metadata_file, 'r') as fp:
        for line in fp:
            cols = line.split('|')
            wav_file = cols[0] + '.wav'
            text = cols[1]
            items.append([text, wav_file])
    return items


def generate_phoneme_sequence(text, phoneme_file):
    phonemes = phoneme_to_sequence(text, ['phoneme_cleaners'],
                                   language='en-us',
                                   enable_eos_bos=False)
    phonemes = np.asarray(phonemes, dtype=np.int32)
    np.save(phoneme_file, phonemes)
    return phonemes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract phonemes and melspectrograms from LJSpecch for training Tacotron')
    parser.add_argument('data_root', type=str, help='Data root directory')
    args = parser.parse_args()

    wav_dir = os.path.join(args.data_root, 'wavs')
    if not os.path.exists(wav_dir):
        raise FileNotFoundError('{} not found'.format(wav_dir))

    metadata_file = os.path.join(args.data_root, 'metadata.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError('{} not found'.format(metadata_file))

    melspec_dir = os.path.join(args.data_root, 'melspec')
    if not os.path.exists(melspec_dir):
        os.makedirs(melspec_dir, exist_ok=True)

    spec_dir = os.path.join(args.data_root, 'spec')
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir, exist_ok=True)

    phoneme_dir = os.path.join(args.data_root, 'phoneme')
    if not os.path.exists(phoneme_dir):
        os.makedirs(phoneme_dir, exist_ok=True)

    items = load_metadata(metadata_file)
    ap = AudioProcessor()

    for text, wav_file in tqdm(items):
        prefix = wav_file.replace('.wav', '')

        # 音素系列を生成
        generate_phoneme_sequence(
            text, os.path.join(phoneme_dir, prefix + '.npy'))

        wav = np.array(ap.load_wav(os.path.join(wav_dir, wav_file)),
                       dtype=np.float32)

        # メルスペクトログラムを生成
        melspec = ap.melspectrogram(wav).astype('float32')
        np.save(os.path.join(melspec_dir, prefix + '.npy'), melspec)

        # 線形スペクトログラムを生成
        spec = ap.spectrogram(wav).astype('float32')
        np.save(os.path.join(spec_dir, prefix + '.npy'), spec)
