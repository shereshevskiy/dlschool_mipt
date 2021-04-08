import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x, temperature=10): # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(hidden)  # !!
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        # #outputs are always from the top hidden layer
        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)

            cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim)
            cell = cell.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)
            
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, temperature=10):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.temperature = temperature
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        
        # repeat hidden and concatenate it with encoder_outputs
        '''your code'''
        hidden = hidden[-1].repeat(encoder_outputs.shape[0], 1, 1)
        # print("hidden.shape", hidden.shape)
        # print("encoder_outputs.shape", encoder_outputs.shape)
        catted = torch.cat((hidden, encoder_outputs), dim=2)
        # calculate energy
        '''your code'''
        # print("hidden.shape", hidden.shape)
        # print("encoder_outputs.shape", encoder_outputs.shape)
        # print("catted.shape", catted.shape)
        # print(self.attn.in_features)
        energy = F.tanh(self.attn(catted))  # [src sent len, batch size, enc_hid_dim]

        # get attention, use softmax function which is defined, can change temperature
        '''your code'''
        attention = self.v(energy)  # [src sent len, batch size, 1]
        attention = softmax(attention, self.temperature).permute(1, 0, 2)  # [batch size, src sent len, 1]
            
        return attention   # '''your code'''
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)  # '''your code'''
        
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, dropout=dropout)   #'''your code''' # use GRU
        
        self.out = nn.Linear(dec_hid_dim + emb_dim + dec_hid_dim, output_dim)   # '''your code''' # linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0) # because only one word, no words sequence 
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        # get weighted sum of encoder_outputs
        #outputs = [src sent len, batch size, hid dim * n directions]
        #attention = [batch size, src sent len, 1]
        '''your code'''
        # print("self.attention(hidden, encoder_outputs)", self.attention(hidden, encoder_outputs).shape)
        # print("encoder_outputs.permute(1, 2, 0)", encoder_outputs.permute(1, 2, 0).shape)
        
        hidden = hidden[-1].unsqueeze(0)
        weights = self.attention(hidden, encoder_outputs)

        context = torch.bmm(weights.permute(0, 2, 1), encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)  # [1, batch size, hid dim * n directions]

        # concatenate weighted sum and embedded, break through the GRU
        '''your code'''
        
        rnn_input = torch.cat([embedded, context], dim=2)  # [1, B, E+H]
        # print("rnn_input.shape", rnn_input.shape)
        # print("hidden.shape", hidden.shape)
        output, hidden = self.rnn(rnn_input, hidden)
        # get predictions
        '''your code'''
        prediction = self.out(torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden  # '''your code'''
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim * 2 == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):

            '''your code'''
            output, hidden = self.decoder(input, hidden, enc_states)

            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1) 
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
