import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional = False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=False,
                            bidirectional=bidirectional,)

    def forward(self, x):
        # x: (seq_len, batch_size, input_size)      seq len would be the number of touchpoints
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (seq_len, batch_size, hidden_size)
        # hidden, cell: (num_layers, batch_size, hidden_size)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size)
        decoder_hidden = decoder_hidden.unsqueeze(2)  # (batch_size, hidden_size, 1)
        encoder_outputs = encoder_outputs.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
        attention_scores = torch.bmm(encoder_outputs, decoder_hidden)  # (batch_size, seq_len, 1)
        attention_scores = F.softmax(attention_scores.squeeze(2), dim=1)  # (batch_size, seq_len)
        return attention_scores

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=False)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Concatenate context and decoder output

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (1, batch_size, output_size) - previous output token
        # hidden, cell: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))  # lstm_out: (1, batch_size, hidden_size)

        # Compute attention scores
        attention_scores = self.attention(lstm_out.squeeze(0), encoder_outputs)  # (batch_size, seq_len)
        attention_scores = attention_scores.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Compute context vector
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        context = torch.bmm(attention_scores, encoder_outputs)  # (batch_size, 1, hidden_size)

        # Concatenate context and LSTM output
        lstm_out = lstm_out.permute(1, 0, 2)  # (batch_size, 1, hidden_size)
        concat_output = torch.cat((lstm_out.squeeze(1), context.squeeze(1)), dim=1)  # (batch_size, hidden_size * 2)

        # Predict next token
        output = self.fc(concat_output)  # (batch_size, output_size)
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        # input: (src_seq_len, batch_size, number of features)
        # target: (word length, batch_size, 1 as its just an index)
        batch_size = target.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.decoder.output_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encode the source sequence
        encoder_outputs, hidden, cell = self.encoder(input)

        # First input to the decoder is the blank token which is just 0
        decoder_input = target[0, :, :].unsqueeze(0)  # (1, batch_size, output_size)

        for t in range(1, trg_len):
            # Decode one token at a time
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[t, :, :] = output  # Store the output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target[t, :, :].unsqueeze(0) if teacher_force else output.unsqueeze(0)

        return outputs
