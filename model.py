import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import random

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional = True):
        """
        LSTM in the model is always batch first
        param input_size: input size for encoder, 6
        param hidden_size: hidden_size for LSTM
        param num_layers: number of layers
        param bidirectional: bool
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=bidirectional,)
        self.fc_hidden = nn.Linear(2*hidden_size,hidden_size)
        self.fc_cell = nn.Linear(2*hidden_size,hidden_size)
        self.fc_out = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, x):
        """
        : param x: (batch_size, seq_len, input_size)      seq len would be the number of touchpoints
        : return outputs: (seq_len, batch_size, hidden_size)
        : return hidden, cell: the final hidden and cell states
        """
        lstm_outputs, (hidden, cell) = self.lstm(x)
        # shape now of states is (2, hidden state) after concat
        hidden_states = torch.cat([torch.cat((hidden[i,:,:], hidden[i+1,:,:]), dim=1).unsqueeze(0) for i in range(0, hidden.shape[0], 2)], dim=0)
        cell_states = torch.cat([torch.cat((cell[i,:,:], cell[i+1,:,:]), dim=1).unsqueeze(0) for i in range(0, cell.shape[0], 2)], dim=0)

        hidden = torch.relu(self.fc_hidden(hidden_states))
        cell = torch.relu(self.fc_cell(cell_states))
        outputs = torch.relu(self.fc_out(lstm_outputs))

        return outputs, hidden, cell
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2, bidirectional=False):
        """
        LSTM in the model is always batch first
        param input_size: input size for decoder, should equal 27 also equals output size
        param hidden_size: hidden_size for LSTM
        param num_layers: number of layers
        param bidirectional: bool
        """
        super(Decoder, self).__init__()
        self.outuput_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=bidirectional,)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, hidden, cell, encoder_output):
        """
        param hidden: previous hidden state of decoder  (num_layers, N, hidden_size)
        param cell: previous cell state of decoder  (num_layers, N, hidden_size)
        param encoder_output: encoder output used for attention value and key
        """

        query = hidden[-1].unsqueeze(0).permute(1, 0, 2) # (N, num_layers, hidden_size)
        context = F.scaled_dot_product_attention(query, encoder_output, encoder_output) # (N, 1, hidden_size)
        # attention weights used as input for lstm
        # lstm_out shape (N, 1, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(context, (hidden, cell))
        output = F.log_softmax(self.fc(lstm_out), dim=-1)
        
        # output: (batch_size, 1, hidden_size)
        # hidden, cell: (num_layers, batch_size, hidden_size)
        return output, hidden, cell    

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size=6, output_size=27, max_letters = 20):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size                # number of features for input
        self.output_size = output_size              # output size of decoder, should be 27
        self.encoder_hidden_size = hidden_size      # hidden size of encoder and decoder
        self.num_layers = num_layers                # number of layers for both lstms
        self.max_letters = max_letters              # max length of prediction

        self.encoder = Encoder(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True)

        self.decoder = Decoder(hidden_size=hidden_size,
                               output_size=output_size,
                               num_layers=num_layers,
                               bidirectional=False) # keep decoder unidirectional
    
    def forward(self, input):
        """
        param input: (N, T, 6) input sequence
        """
        batch_size = input.shape[0]

        # encode the sequence
        encoder_output, hidden, cell = self.encoder(input)

        # where to store all the log probabilities
        outputs = torch.zeros(batch_size, encoder_output.shape[1], self.output_size).to(next(self.parameters()).device)

        for i in range(encoder_output.shape[1]):
            output, hidden, cell = self.decoder(hidden, cell, encoder_output)
            # print(output.shape)
            outputs[:,i,:] = output.squeeze(1)

        return outputs