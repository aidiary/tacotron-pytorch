import argparse
import os
import numpy as np
import torch
from model import Tacotron
from utils.text import phoneme_to_sequence
from utils.text.symbols import phonemes
from utils.audio import AudioProcessor


parser = argparse.ArgumentParser()
parser.add_argument('text', type=str, help='text to generate speech')
parser.add_argument('model_path', type=str, help='path to model file')
parser.add_argument('out_path', type=str,
                    help='path to save final wav file')
parser.add_argument('--gpu_id', type=str, default=0, help='gpu_id')
args = parser.parse_args()
print(args)

device = torch.device('cuda:{}'.format(args.gpu_id)
                      if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)


def main():
    ap = AudioProcessor()

    # load model
    num_chars = len(phonemes)
    model = Tacotron(num_chars).to(device)
    cp = torch.load(args.model_path)
    model.load_state_dict(cp['model'])
    model.eval()

    print('Text: {}'.format(args.text))
    wav = tts(model, args.text, ap)

    file_name = args.text.replace(' ', '_') + '.wav'
    out_path = os.path.join(args.out_path, file_name)
    ap.save_wav(wav, out_path)


def tts(model, text, ap):
    inputs = phoneme_to_sequence(text,
                                 ['phoneme_cleaners'],
                                 language='en-us',
                                 enable_eos_bos=False)
    inputs = np.asarray(inputs, dtype=np.int32)
    inputs = torch.from_numpy(inputs).unsqueeze(0).long().to(device)

    decoder_output, postnet_output, alignments, stop_tokens = \
        model.inference(inputs)
    postnet_output = postnet_output[0].data.cpu().numpy()
    wav = ap.inv_spectrogram(postnet_output.T)
    return wav


if __name__ == "__main__":
    main()
