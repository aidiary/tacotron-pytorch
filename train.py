import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from dataset import TTSDataset, preprocess_ljspeech
from model import Tacotron
from loss import L1LossMasked
from utils.utils import count_parameters
from utils.audio import AudioProcessor
from utils.text.symbols import phonemes


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)

gpu_id = 3
r = 7
batch_size = 32
lr = 0.0001
epochs = 1000
out_path = 'runs'

device = torch.device('cuda:{}'.format(gpu_id)
                      if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    ap = AudioProcessor()

    # Load dataset
    train_items = preprocess_ljspeech(
        'data/LJSpeech-1.1', 'metadata_train.csv')
    valid_items = preprocess_ljspeech(
        'data/LJSpeech-1.1', 'metadata_val.csv')

    train_dataset = TTSDataset(
        7,
        'phoneme_cleaners',
        ap,
        train_items,
        phoneme_cache_path='ljspeech_phonemes')

    valid_dataset = TTSDataset(
        7,
        'phoneme_cleaners',
        ap,
        valid_items,
        phoneme_cache_path='ljspeech_phonemes')

    print('train data:', len(train_dataset))
    print('valid data:', len(valid_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=False)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=False)

    # Create models
    num_chars = len(phonemes)
    model = Tacotron(num_chars).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    # StopNetは二値分類タスクなので独自に訓練する
    optimizer_st = optim.Adam(
        model.decoder.stopnet.parameters(), lr=lr, weight_decay=0.0)

    criterion = L1LossMasked()
    criterion_st = nn.BCEWithLogitsLoss()

    num_params = count_parameters(model)
    print('Model has {} parameters'.format(num_params))

    # Training
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train(train_loader, model,
                           criterion, criterion_st,
                           optimizer, optimizer_st, ap, epoch)
        valid_loss = evaluate(valid_loader, model,
                              criterion, criterion_st, ap, epoch)
        print('Epoch [{}/{}] train_loss: {:.5f} valid_loss: {:.5f}'
              .format(epoch, epochs, train_loss, valid_loss))

        if valid_loss < best_loss:
            print('  => valid_loss improved from {:.5f} to {:.5f}!'
                  .format(best_loss, valid_loss))
            new_state_dict = model.state_dict()
            state = {
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'linear_loss': valid_loss
            }
            best_loss = valid_loss
            best_model_path = os.path.join(out_path, 'best_model.pth.tar')


def train(train_loader, model, criterion, criterion_st,
          optimizer, optimizer_st, ap, epoch):
    model.train()
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_stop_loss = 0
    batch_n_iter = int(len(train_loader.dataset) / batch_size)

    for num_iter, data in enumerate(train_loader):
        text_input = data[0].to(device)
        text_lengths = data[1].to(device)
        linear_input = data[2].to(device)
        mel_input = data[3].to(device)
        mel_lengths = data[4].to(device)
        stop_targets = data[5].to(device)

        # rフレームずつまとめる
        # そのrフレームに1つでも1（停止）が入っていれば停止とみなす
        stop_targets = stop_targets.view(
            text_input.shape[0],
            stop_targets.size(1) // r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(
            2).float().squeeze(2)

        optimizer.zero_grad()
        optimizer_st.zero_grad()

        decoder_output, postnet_output, alignments, stop_tokens = model(
            text_input, text_lengths, mel_input)
        print(decoder_output.shape)
        print(postnet_output.shape)
        print(alignments.shape)
        print(stop_tokens.shape)


def evaluate(valid_loader, model, criterion, criterion_st, ap, epoch):
    pass


if __name__ == "__main__":
    main()
