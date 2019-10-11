import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.attention_weights = torch.zeros(
            [B, T], dtype=torch.float32, device=inputs.device)
        self.attention_weights_cum = torch.zeros(
            [B, T], dtype=torch.float32, device=inputs.device)

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
