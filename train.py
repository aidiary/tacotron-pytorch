import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from dataset import TTSDataset
from model import Tacotron
from loss import L1LossMasked
from utils.utils import count_parameters
from utils.audio import AudioProcessor
from utils.text.symbols import phonemes

from tqdm import tqdm
from tensorboardX import SummaryWriter


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)

gpu_id = 3
r = 7
batch_size = 32
lr = 0.0001
epochs = 1000
logging_step = 10
checkpoint_step = 10000

device = torch.device('cuda:{}'.format(gpu_id)
                      if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


def main():
    ap = AudioProcessor()

    train_dataset = TTSDataset(
        'data/LJSpeech-1.1', 'train.list', outputs_per_step=7)
    valid_dataset = TTSDataset(
        'data/LJSpeech-1.1', 'valid.list', outputs_per_step=7)

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
        valid_dataset,
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
    global_step = 0
    for epoch in range(0, epochs + 1):
        train_loss, global_step = train(train_loader, model,
                                        criterion, criterion_st,
                                        optimizer, optimizer_st, ap,
                                        global_step, epoch)

        valid_loss = evaluate(valid_loader, model,
                              criterion, criterion_st, ap,
                              global_step, epoch)

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
            best_model_path = os.path.join(writer.logdir, 'best_model.pth')


def train(train_loader, model, criterion, criterion_st,
          optimizer, optimizer_st, ap, global_step, epoch):
    model.train()
    avg_decoder_loss = 0
    avg_postnet_loss = 0
    avg_stop_loss = 0
    batch_n_iter = int(len(train_loader.dataset) / batch_size)

    for data in tqdm(train_loader, total=len(train_loader), desc='train'):
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
        stop_targets = (stop_targets.sum(2) > 0.0)  \
            .unsqueeze(2).float().squeeze(2)

        global_step += 1

        optimizer.zero_grad()
        optimizer_st.zero_grad()

        # forward
        decoder_output, postnet_output, alignments, stop_tokens = model(
            text_input, text_lengths, mel_input)

        decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
        postnet_loss = criterion(postnet_output, linear_input, mel_lengths)
        loss = decoder_loss + postnet_loss
        loss.backward()
        optimizer.step()

        # 生成を停止するか判断するネットワークの訓練
        stop_loss = criterion_st(stop_tokens, stop_targets)
        stop_loss.backward()
        optimizer_st.step()

        avg_decoder_loss += float(decoder_loss.item())
        avg_postnet_loss += float(postnet_loss.item())
        avg_stop_loss += float(stop_loss.item())

        # logging_stepごとにTensorboardにlossを記録
        if global_step % logging_step == 0:
            writer.add_scalar('train/postnet_loss',
                              postnet_loss.item(), global_step)
            writer.add_scalar('train/decoder_loss',
                              decoder_loss.item(), global_step)
            writer.add_scalar('train/stop_loss',
                              stop_loss.item(), global_step)

        # checkpoint_stepごとにcheckpointを保存
        if global_step % checkpoint_step == 0:
            checkpoint_path = os.path.join(
                writer.logdir, 'checkpoint_{}.pth'.format(global_step))
            state_dict = model.state_dict()
            state = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'optimizer_st': optimizer_st.state_dict(),
                'step': global_step,
                'epoch': epoch,
                'linear_loss': postnet_loss.item()
            }
            torch.save(state, checkpoint_path)

    num_batch = len(train_loader)
    avg_decoder_loss /= num_batch
    avg_postnet_loss /= num_batch
    avg_stop_loss /= num_batch

    return avg_postnet_loss, global_step


def evaluate(valid_loader, model, criterion, criterion_st, ap, global_step, epoch):
    model.eval()
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_stop_loss = 0

    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader), desc='evaluate'):
            text_input = data[0].to(device)
            text_lengths = data[1].to(device)
            linear_input = data[2].to(device)
            mel_input = data[3].to(device)
            mel_lengths = data[4].to(device)
            stop_targets = data[5].to(device)

            stop_targets = stop_targets.view(
                text_input.shape[0],
                stop_targets.size(1) // r, -1)
            stop_targets = (stop_targets.sum(2) > 0.0)  \
                .unsqueeze(2).float().squeeze(2)

            # forward
            decoder_output, postnet_output, alignments, stop_tokens = model(
                text_input, text_lengths, mel_input)

            decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
            postnet_loss = criterion(postnet_output, linear_input, mel_lengths)
            loss = decoder_loss + postnet_loss

            stop_loss = criterion_st(stop_tokens, stop_targets)

            avg_decoder_loss += float(decoder_loss.item())
            avg_postnet_loss += float(postnet_loss.item())
            avg_stop_loss += float(stop_loss.item())

        num_batch = len(valid_loader)
        avg_decoder_loss /= num_batch
        avg_postnet_loss /= num_batch
        avg_stop_loss /= num_batch

        # evaluateはepochごとにlossを記録
        writer.add_scalar('valid/postnet_loss',
                          avg_postnet_loss.item(), global_step)
        writer.add_scalar('valid/decoder_loss',
                          avg_decoder_loss.item(), global_step)
        writer.add_scalar('valid/stop_loss',
                          avg_stop_loss.item(), global_step)

    return avg_postnet_loss


if __name__ == "__main__":
    main()
