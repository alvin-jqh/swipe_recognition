import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional = False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=bidirectional,)
        self.fc = nn.Linear(2*hidden_size,hidden_size)

    def forward(self, x):
        # x: (seq_len, batch_size, input_size)      seq len would be the number of touchpoints
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (seq_len, batch_size, hidden_size)
        # hidden, cell: (num_layers, batch_size, hidden_size)
        return outputs, hidden, cell
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_letters=10, force_ratio = 0.7, num_layers = 2, bidirectional = False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.force_ratio = force_ratio
        self.max_letters=max_letters

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=2*hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        # the output size doubles if the lstm is bidirectional
        fc_in_size = 2*hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_size,output_size)

    def forward(self, key, value, encoder_hidden, encoder_cell, word_tensor=None):
        """
        Args:
            key: key matrix derived from output from encoder (N, L, decoder hidden)
            value: value matrix derived output from encoder (N, L, decoder hidden)
            encoder_hidden: hidden state from encoder (D * Layers, N, Hidden) for encoder
            encoder_cell: cell state from encoder (D * Layers, N, Hidden) for encoder
            word_tensor: word tensors containing indicies of letters (N, max length)
        """
        batch_size = key.shape[0]
        # maximum word length to output, caps out at 10
        word_length = min(self.max_letters, word_tensor.shape[-1]) if word_tensor is not None else self.max_letters
        # create the first decoder input, which is blank inputs
        decoder_input = torch.tensor([0] * batch_size, dtype=torch.long, device=next(self.parameters()).device)
        # initial hidden and cell state comes from final layer of encoder, shape is (N, encoder hidden)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoder_outputs = torch.Tensor().to(next(self.parameters()).device)
        # for each letter in word
        for i in range(word_length):
            decoder_output, decoder_hidden, decoder_cell = self.step(decoder_input,
                                                                     decoder_hidden,
                                                                     decoder_cell,
                                                                     key,
                                                                     value)
            decoder_outputs = torch.cat((decoder_outputs,decoder_output), dim=1)  # add to list of outputs
            # print(decoder_output.shape)

            teacher_force = torch.rand(1).item() < self.force_ratio
            
            if teacher_force and word_tensor is not None:
                decoder_input = word_tensor[:,i]    # next letter
            else:
                # for using when not teacher forcing, use model prediction
                decoder_input = torch.argmax(decoder_output.squeeze(1), dim=-1) # 1D with length N
        
        probs = F.softmax(decoder_outputs, dim=-1)

        return probs, decoder_hidden, decoder_cell

    def step(self, decoder_input, hidden, cell, key, value):
        """
        Args:
            decoder_input: shape (N), should be indicies for the letters, dtype either int or long
            hidden: hidden state of lstm (D*Layers, N, decoder hidden)
            cell: cell state of lstm (D*Layers, N, deocder hidden)
            key: key matrix derived from output from encoder (N, L, decoder hidden)
            value: value matrix derived output from encoder (N, L, decoder hidden)
        """
        embedded = self.embedding(decoder_input.unsqueeze(1))  # (N, 1 decoder hidden)
        query = hidden[-1].unsqueeze(0).permute(1, 0, 2) # (N, D*Layers, decoder hidden)    Query should come from the decoders hidden state
        # print(query.shape)
        context = F.scaled_dot_product_attention(query, key, value) # (N, 1, decoder hidden)
        # print(context.shape)
        input_lstm = torch.cat((embedded, context), dim=-1) # (N, 1, 2*decoder hidden)
        output_lstm, (hidden, cell) = self.lstm(input_lstm, (hidden, cell)) # output lstm (N, 1, D*hidden)
        output_fc = self.fc(output_lstm)

        return output_fc, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional=False, input_size=6, max_letters=10, force_ratio=0.7, output_size=27):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size                # number of features for input
        self.output_size = output_size              # output size of decoder, should be 27
        self.encoder_hidden_size = hidden_size      # hidden size of encoder and decoder
        self.num_layers = num_layers                # number of layers for both lstms
        self.bidirectional = bidirectional          # bidirectional for both lstms
        self.max_letters = max_letters              # max length for word output, word tensors should be padded to this length
        self.force_ratio = force_ratio              # chance for teacher forcing in training

        self.encoder = Encoder(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional)

        self.decoder = Decoder(hidden_size=hidden_size,
                               output_size=output_size,
                               max_letters=max_letters,
                               force_ratio=force_ratio,
                               num_layers=num_layers,
                               bidirectional=False) # keep decoder unidirectional
        
        # used for projecting the hidden and cell for both layers of the encoder to the inital decoder state
        self.hidden_projection_1 = nn.Linear(hidden_size*2,hidden_size)
        self.cell_projection_1 = nn.Linear(hidden_size*2,hidden_size)

        self.hidden_projection_2 = nn.Linear(hidden_size*2,hidden_size)
        self.cell_projection_2 = nn.Linear(hidden_size*2,hidden_size)

        self.key_projection = nn.Linear(hidden_size*2,hidden_size)
        self.value_projection = nn.Linear(hidden_size*2,hidden_size)
    
    def forward(self, input, word_tensors=None):
        # calculate the encoder outputs
        encoder_output, encoder_hidden, encoder_cell = self.encoder(input)
        # if the encoder is bidirectional, the hidden and cell states of the encoder need to be projected to match dimensions
        if self.bidirectional:
            # concatenate the layer states 
            layer_1_hidden = torch.cat((encoder_hidden[0], encoder_hidden[2]), dim=-1)
            layer_2_hidden = torch.cat((encoder_hidden[1], encoder_hidden[3]), dim=-1)

            layer_1_cell = torch.cat((encoder_cell[0], encoder_cell[2]), dim=-1)
            layer_2_cell = torch.cat((encoder_cell[1], encoder_cell[3]), dim=-1)

            # project
            project_h1 = F.softmax(self.hidden_projection_1(layer_1_hidden), dim=-1).unsqueeze(0)
            project_h2 = F.softmax(self.hidden_projection_2(layer_2_hidden), dim=-1).unsqueeze(0)

            project_c1 = F.softmax(self.cell_projection_1(layer_1_cell), dim=-1).unsqueeze(0)
            project_c2 = F.softmax(self.cell_projection_2(layer_2_cell), dim=-1).unsqueeze(0)

            # recombine
            encoder_hidden = torch.cat((project_h1, project_h2), dim=0)
            encoder_cell = torch.cat((project_c1, project_c2), dim=0)

            key = self.key_projection(encoder_output)
            value = self.value_projection(encoder_output)
        else:
            # if not bidirectional, key and value arent transformed
            key = encoder_output
            value = encoder_output

        # print(encoder_hidden.shape)
        # print(encoder_cell.shape)
        # print(encoder_output.shape)
        # decoder
        log_probs, decoder_hidden, decoder_cell = self.decoder(key, value, encoder_hidden, encoder_cell, word_tensors)

        return log_probs  
     
     
