import torch
import torch.nn as nn
import torch.nn.functional as F


class Tacotron(nn.Module):

    def __init__(self, num_chars, r=5, linear_dim=1025, mel_dim=80):
        super(Tacotron, self).__init__()
        self.r = r
        self.linear_dim = linear_dim
        self.mel_dim = mel_dim

        self.embedding = nn.Embedding(num_chars, 256)
        self.embedding.weight.data.normal_(0, 0.3)

        self.encoder = Encoder(256)
        self.decoder = Decoder(256, mel_dim, r)


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
    pass


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units

        Args:
            in_features (int): sample size
            K (int): max filter size in conv bank
            projections (list): conv channel sizes for conv projections
            num_highways (int): number of highways layers

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
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
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
