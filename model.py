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
        self.lstm = nn.LSTM(input_size=output_size+hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=bidirectional,)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, input, hidden, cell, encoder_output):
        """
        param input: (batch_size, 1, 27) next input, either previous output or one hot encoded letters
        param hidden: previous hidden state of decoder  (num_layers, N, hidden_size)
        param cell: previous cell state of decoder  (num_layers, N, hidden_size)
        param encoder_output: encoder output used for attention value and key
        """

        query = hidden[-1].unsqueeze(0).permute(1, 0, 2) # (N, num_layers, hidden_size)
        context = F.scaled_dot_product_attention(query, encoder_output, encoder_output)
        print(context.shape)
        input = torch.cat((input, context), dim = -1)   # (N, 1, hidden + output_size)

        # lstm_out shape (N, 1, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(input, (hidden, cell))
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
    
    def forward(self, input, target_tensor, force_ratio = 0.5):
        """
        param input: (N, T, 6) input sequence
        param target_tensor: (N, max_word_length) padded word tensors
        param force_ratio: chance for teacher forcing
        """
        batch_size = input.shape[0]
        target_length = target_tensor.shape[1]
        # where to store all the log probabilities
        outputs = torch.zeros(batch_size, target_length, self.output_size).to(next(self.parameters()).device)

        # encode the sequence
        encoder_output, hidden, cell = self.encoder(input)
        # get the first decoder input
        blank = torch.LongTensor([0]*batch_size).to(next(self.parameters()).device)
        decoder_input = F.one_hot(blank, num_classes=27).unsqueeze(1)   # (N, 1, 27)

        for i in range(target_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_output)
            print(output.shape)
            outputs[:,i,:] = output.squeeze(1)

            if i < target_length - 1:
                teacher_force = torch.rand(1).item() < force_ratio  # probability of being true or false
                if teacher_force:
                    decoder_input = F.one_hot(target_tensor[:, i], num_classes=27).unsqueeze(1)
                else:
                    decoder_input = output

        return outputs
    
    def predict(self, input, max_length):
        batch_size = input.shape[0]
        # store outputs
        outputs = torch.zeros(batch_size, max_length, self.output_size).to(next(self.parameters()).device)

        # encode the sequence
        encoder_output, hidden, cell = self.encoder(input)
        # get the first decoder input
        blank = torch.LongTensor([0]*batch_size).to(next(self.parameters()).device)
        decoder_input = F.one_hot(blank, num_classes=27).unsqueeze(1)   # (N, 1, 27)

        for i in range(max_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_output)
            outputs[:,i,:] = output.squeeze(1)
            decoder_input = output

        return outputs