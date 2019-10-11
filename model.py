import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import CBHG, BatchNormConv1d, Linear, Prenet, Highway, Attention, LocationLayer, StopNet
from utils.utils import sequence_mask


class Tacotron(nn.Module):

    def __init__(self, num_chars, r=7, linear_dim=1025, mel_dim=80):
        super(Tacotron, self).__init__()
        self.r = r
        self.linear_dim = linear_dim
        self.mel_dim = mel_dim

        self.embedding = nn.Embedding(num_chars, 256)
        self.embedding.weight.data.normal_(0, 0.3)

        self.encoder = Encoder(256)
        self.decoder = Decoder(256, mel_dim, r)
        self.postnet = CBHG(
            mel_dim,
            K=8,
            conv_bank_features=128,
            conv_projections=[256, mel_dim],
            highway_features=128,
            gru_features=128,
            num_highways=4)
        self.last_linear = nn.Linear(
            self.postnet.gru_features * 2, linear_dim)

    def forward(self, characters, text_lengths, mel_specs):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)

        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        # 複数フレームがまとまっているので元に戻す
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        # メルスペクトログラムを線形スペクトログラムに変換する
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference(self, characters):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        mel_outputs, alignments, stop_tokens = \
            self.decoder.inference(encoder_outputs)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens


class Encoder(nn.Module):

    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_features, out_features=[256, 128])
        self.cbhg = CBHG(128,
                         K=16,
                         conv_bank_features=128,
                         conv_projections=[128, 128],
                         highway_features=128,
                         gru_features=128,
                         num_highways=4)

    def forward(self, inputs):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)


class Decoder(nn.Module):

    def __init__(self, in_features, memory_dim, r=7):
        super(Decoder, self).__init__()

        # Encoderの出力の次元数
        self.in_features = in_features
        # 1つ前の時刻のDecoderの出力の次元数（rフレーム分）
        self.memory_dim = memory_dim
        # decodeの各ステップでrフレーム分をまとめて出力する
        self.memory_size = r
        # rフレーム分をまとめて処理するが訓練が進むに連れて短くなる
        self.r_init = r
        self.r = r
        self.max_decoder_steps = 500
        self.query_dim = 256

        self.prenet = Prenet(
            self.memory_dim,
            out_features=[256, 128])

        self.attention_rnn = nn.GRUCell(in_features + 128, self.query_dim)

        self.attention = Attention(query_dim=self.query_dim,
                                   embedding_dim=in_features,
                                   attention_dim=128)

        # (processed_memory | attention context) -> |Linear| -> decoder RNN input
        self.project_to_decoder_in = nn.Linear(256 + in_features, 256)
        # decoder RNN input -> |RNN| -> RNN state
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)]
        )
        # RNN state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(256, self.memory_dim * self.r_init)
        self.stopnet = StopNet(256 + memory_dim * self.r_init)

    def forward(self, inputs, memory, mask):
        """
        Args:
            inputs: Encoder outputs.
            memory: Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.
            mask: Attention mask for sequence padding.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        # 入力となるメルスペクトログラム系列をrフレームごとにまとめて特徴量とみなす変換
        memory = self._reshape_memory(memory)

        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        self.attention.init_states(inputs)
        # 訓練なので教師のメルスペクトログラムの系列長まで生成
        while len(outputs) < memory.size(0):
            if t > 0:
                # 0フレーム目は0初期化
                # 1つ前のフレームのメルスペクトログラムが入る
                new_memory = memory[t - 1]
                self._update_memory_input(new_memory)
            output, stop_token, attention = self.decode(inputs, mask)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1

        return self._parse_outputs(outputs, attentions, stop_tokens)

    def decode(self, inputs, mask=None):
        """
        inputsはEncoderの出力

        decode()は2つのRNNが連結してできている
        decode()は1ステップ分の処理なのでGRUCellでできている
        1. Attention RNN => attention_rnn_hidden
        2. Decoder RNN   => decoder_rnn_hiddens[]
        それぞれ内部状態を持ち各ステップの間で保持している
        Decoder RNNは2層GRUになっている
        """
        # Prenet
        processed_memory = self.prenet(self.memory_input)

        # Attention RNN
        # hiddenが循環していることに注意！
        # 次のstepでは更新されたhiddenが使われる
        self.attention_rnn_hidden = self.attention_rnn(
            torch.cat((processed_memory, self.context_vec), -1),
            self.attention_rnn_hidden)

        # Attention
        # AttentionはAttention RNNの隠れ状態とEncoder出力の間で行われる
        # processed_inputsはinputsをLinearに通してサイズ変更したtensor
        # TODO: processed_inputsは外から渡さなくてもattentionの中で計算できそう
        self.context_vec = self.attention(
            self.attention_rnn_hidden, inputs, self.processed_inputs, mask)

        # Decoder RNN
        # Attention RNNのhiddenとcontext vectorを入力する
        decoder_input = self.project_to_decoder_in(
            torch.cat((self.attention_rnn_hidden, self.context_vec), -1))

        # Decoder RNNは複数のGRU層からなる
        # 隠層で異なる隠れ状態をもつ
        for idx in range(len(self.decoder_rnns)):
            self.decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                decoder_input, self.decoder_rnn_hiddens[idx])
            # 次の層のRNNへは入力を足しこむ
            # Residual connection
            decode_input = self.decoder_rnn_hiddens[idx] + decoder_input
        decoder_output = decoder_input

        # Decoderの出力からメルスペクトログラムを予測
        output = self.proj_to_mel(decoder_output)

        # Stop tokenを予測
        # このステップで停止するかを予測するネットワーク
        stopnet_input = torch.cat([decoder_output, output], -1)
        stop_token = self.stopnet(stopnet_input.detach())
        output = output[:, :self.r * self.memory_dim]
        return output, stop_token, self.attention.attention_weights

    def inference(self, inputs):
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        self.attention.init_states(inputs)
        while True:
            if t > 0:
                new_memory = outputs[-1]
                self._update_memory_input(new_memory)
            output, stop_token, attention = self.decode(inputs, None)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1
            # TODO: 1つ目の条件は何を意味する？
            if t > inputs.shape[1] / 4 and (stop_token > 0.6 or attention[:, -1].item() > 0.6):
                break
            elif t > self.max_decoder_steps:
                print('WARNING: Decoder stopped with max_decoder_steps')
                break
        return self._parse_outputs(outputs, attentions, stop_tokens)

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        B = memory.shape[0]
        # rフレームずつをまとめて特徴量とみなす
        if memory.size(-1) == self.memory_dim:
            memory = memory.contiguous()
            memory = memory.view(B, memory.size(1) // self.r, -1)
        # (seq_len, batch, input_size) に変換
        memory = memory.transpose(0, 1)
        return memory

    def _update_memory_input(self, new_memory):
        self.memory_input = new_memory[:, :self.memory_dim]

    def _init_states(self, inputs):
        """
        Initialize decoder states
        """
        B = inputs.size(0)
        T = inputs.size(1)

        self.memory_input = torch.zeros(
            B, self.memory_dim, device=inputs.device)
        self.attention_rnn_hidden = torch.zeros(B, 256, device=inputs.device)
        self.decoder_rnn_hiddens = [
            torch.zeros(B, 256, device=inputs.device) for _ in range(len(self.decoder_rnns))
        ]
        self.context_vec = inputs.data.new(B, self.in_features).zero_()
        self.processed_inputs = self.attention.inputs_layer(inputs)

    def _parse_outputs(self, outputs, attentions, stop_tokens):
        # outputsなどは (batch, input_size) で長さが時間長のリストになっている
        # 時間方向にstackした後に (batch, seq_len, input_size) に戻す
        attentions = torch.stack(attentions).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).squeeze(-1)
        return outputs, attentions, stop_tokens


if __name__ == "__main__":
    characters = torch.ones([32, 71], dtype=torch.long)
    text_lengths = torch.ones(32, dtype=torch.long)
    mel_specs = torch.rand(32, 231, 80)
    tacotron = Tacotron(num_chars=71, r=7, linear_dim=1025, mel_dim=80)
    mel_outputs, linear_outputs, alignments, stop_tokens = tacotron(
        characters, text_lengths, mel_specs)
    print(mel_outputs.shape)
    print(linear_outputs.shape)
    print(alignments.shape)
    print(stop_tokens.shape)
