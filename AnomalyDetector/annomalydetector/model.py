import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length):
        super(Autoencoder, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden_n, _) = self.encoder(x)
        decoder_input = hidden_n.repeat(self.seq_length, 1, 1).permute(1, 0, 2)
        decoder_output, _ = self.decoder(decoder_input)
        return decoder_output
