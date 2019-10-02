import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask = self._sequence_mask(text_lengths).to(characters.device)

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

    def _sequence_mask(self, sequence_length, maxlen=None):
        if maxlen is None:
            maxlen = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, maxlen).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).
                             expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand


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

    # TODO: inferenceを追加


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units

        Args:
            in_features(int): sample size
            K(int): max filter size in conv bank
            projections(list): conv channel sizes for conv projections
            num_highways(int): number of highways layers

        Shapes:
            - input: batch x time x dim
            - output: batch x time x dim*2
    """

    def __init__(self,
                 in_features,
                 K=16,
                 conv_bank_features=128,
                 conv_projections=[128, 128],
                 highway_features=128,
                 gru_features=128,
                 num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.conv_bank_features = conv_bank_features
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.conv_projections = conv_projections
        self.relu = nn.ReLU()

        # conv1d bank with filter size k=1...K
        self.conv1d_banks = nn.ModuleList([
            BatchNormConv1d(
                in_features,
                conv_bank_features,
                kernel_size=k,
                stride=1,
                padding=[(k - 1) // 2, k // 2],
                activation=self.relu) for k in range(1, K + 1)
        ])

        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]

        # conv1d projection layers
        layer_set = []
        for in_size, out_size, act in zip(out_features,
                                          conv_projections,
                                          activations):
            layer = BatchNormConv1d(
                in_size,
                out_size,
                kernel_size=3,
                stride=1,
                padding=[1, 1],
                activation=act)
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)

        # 論文の実装だとPostCBHGのConv1d projectionの出力が80ユニットなので
        # Highwayの128ユニットに入力できないためLinearで合わせる
        if self.highway_features != conv_projections[-1]:
            self.pre_highway = nn.Linear(
                conv_projections[-1], highway_features, bias=False)

        # Highway layers
        self.highways = nn.ModuleList([
            Highway(highway_features, highway_features)
            for _ in range(num_highways)
        ])

        # bi-directional GRU
        self.gru = nn.GRU(
            gru_features,
            gru_features,
            1,
            batch_first=True,
            bidirectional=True)

    def forward(self, inputs):
        # inputs: (batch, seq_len, input_size)
        # (batch, input_size, seq_len)
        x = inputs.transpose(1, 2)

        # Conv1D bank
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)

        # TODO: 論文の実装だとここにMaxPooling1Dが入る？

        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)

        # Conv1D projections
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        # (batch, seq_len, input_size)
        x = x.transpose(1, 2)

        # Residual connection
        x += inputs

        # Highway network
        # PostCBHGではサイズが異なるのでLinearを挟む
        if self.highway_features != self.conv_projections[-1]:
            x = self.pre_highway(x)

        for highway in self.highways:
            x = highway(x)

        # (batch, seq_len, input_size)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs


class BatchNormConv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.padding = padding
        self.padder = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Linear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(
            in_features, out_features, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Prenet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=[256, 256]):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = nn.ModuleList([
            Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5)
        return x


class Highway(nn.Module):

    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Attention(nn.Module):

    def __init__(self, query_dim, embedding_dim, attention_dim):
        super(Attention, self).__init__()
        self.query_layer = Linear(
            query_dim, attention_dim, bias=False, init_gain='tanh')
        self.inputs_layer = Linear(
            embedding_dim, attention_dim, bias=False, init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=True)
        self.location_layer = LocationLayer(attention_dim)
        self._mask_value = -float('inf')

    def forward(self, query, inputs, processed_inputs, mask):
        # Location-sensitive attention
        attention, _ = self.get_location_attention(query, processed_inputs)
        if mask is not None:
            attention.data.masked_fill_(~mask, self._mask_value)
        alignment = torch.sigmoid(attention) / \
            torch.sigmoid(attention).sum(dim=1, keepdim=True)
        self.update_location_attention(alignment)

        # context vector
        context = torch.bmm(alignment.unsqueeze(1), inputs)
        context = context.squeeze(1)

        # current stepのattention重みをインスタンスとして保持しておく
        # location-sensitive
        self.attention_weights = alignment

        return context

    def init_states(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights = torch.zeros([B, T], dtype=torch.float32)
        self.attention_weights_cum = torch.zeros([B, T], dtype=torch.float32)

    def get_location_attention(self, query, processed_inputs):
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
                                   self.attention_weights_cum.unsqueeze(1)), dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(
            torch.tanh(processed_query +
                       processed_attention_weights + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def update_location_attention(self, alignment):
        self.attention_weights_cum += alignment


class LocationLayer(nn.Module):
    def __init__(self,
                 attention_dim,
                 attention_n_filters=32,
                 attention_kernel_size=31):
        super(LocationLayer, self).__init__()
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False)
        self.location_dense = Linear(
            attention_n_filters, attention_dim, bias=False, init_gain='tanh')

    def forward(self, attention_cat):
        processed_attention = self.location_conv(attention_cat)
        processed_attention = self.location_dense(
            processed_attention.transpose(1, 2))
        return processed_attention


class StopNet(nn.Module):

    def __init__(self, in_features):
        super(StopNet, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features, 1)
        torch.nn.init.xavier_uniform_(
            self.linear.weight, gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, inputs):
        # TODO: dropoutが先であっている？
        outputs = self.dropout(inputs)
        outputs = self.linear(outputs)
        return outputs


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
