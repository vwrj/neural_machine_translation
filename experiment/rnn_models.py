import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):

    def __init__(self, args, padding_idx, src_vocab_size):
        super(Encoder, self).__init__()
        self.args = args
        self.num_directions = 2 if args.bidirectional else 1

        self.embedding = nn.Embedding(src_vocab_size, args.embedding_size, padding_idx = padding_idx)
        self.rnn = nn.GRU(
                input_size = args.embedding_size,
                hidden_size = args.hidden_size,
                num_layers = args.num_encoder_layers,
                #dropout = args.dropout,
                bidirectional = args.bidirectional
                )

    def forward(self, hidden, x, lengths):
        # x.shape: Size([32, 16]) 
        # dimension of x: (seq_len, batch, input_size)
        x = self.embedding(x)
        # x.shape: Size([32, 16, 256])
        # dimension of x after embedding: (seq_len, batch, embedding_size)

        x = pack_padded_sequence(x, lengths)
        x, hidden = self.rnn(x, hidden)
        x, output_lengths = pad_packed_sequence(x)
        
        print(x.shape)
        # x.shape: Size([32, 16, 128])
        # dimension of x after encoder: (seq_len, batch, hidden_size)
        # hidden.shape: Size([1, 16, 128])

        if self.num_directions == 2:
            x = x[:, :, :self.args.hidden_size] + x[:, :, self.args.hidden_size:]
        return x, hidden

    def random_init_hidden(self, device, current_batch_size):
        # only needed for encoder, since decoder's first hidden state is the output of encoder

        hidden = torch.zeros(
                self.args.num_encoder_layers * self.num_directions,
                current_batch_size,
                self.args.hidden_size,
                device=device
                )

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        # TODO: try not doing the below, just doing zeros
        nn.init.xavier_normal_(hidden)

        return hidden


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):

        energy = self.score(hidden, encoder_outputs)
        score = F.softmax(energy, dim = 1).view(1, self.batch_size, -1) # works, but bad code. 
        context_vector = torch.bmm(score.transpose(1, 0), encoder_outputs.transpose(1, 0))
        return context_vector, score

    def score(self, hidden, encoder_output):
        '''
        Args
            hidden: size 1 x B x hidden_size
            encoder_output: size N x B x hidden_size
        Return
            energy: size B x N x 1

        torch.bmm performs a batch matrix-matrix product. 
        torch.bmm(batch1, batch2) 
        if batch1 is (B x N x M) and batch2 is (B x M x P), then
        output will be (B x N x P). 
        '''
        self.batch_size = hidden.shape[1]
        if self.method == 'dot':
            energy = torch.bmm(encoder_output.transpose(1, 0), hidden.squeeze(0).unsqueeze(2))
            return energy

        elif self.method == 'general':
            energy = torch.bmm(encoder_output.transpose(1, 0), self.attn(hidden.squeeze(0)).unsqueeze(2))
            return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, args, trg_padding_idx, output_size, device=None):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = args.attn_model
        self.hidden_size = args.hidden_size
        self.output_size = output_size
        self.n_layers = args.num_decoder_layers
        self.embedding_size = args.embedding_size
        self.device = device

        # Define layers
        self.embedding = nn.Embedding(
                self.output_size,
                args.embedding_size, 
                padding_idx = trg_padding_idx
                )

        self.gru = nn.GRU(
                input_size = args.embedding_size,
                hidden_size = args.hidden_size,
                num_layers = self.n_layers
                )

        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.attn = Attn(self.attn_model, self.hidden_size)

    def forward(self, hidden, input_seq, encoder_outputs, src_lengths=None):

        # Get embedding of current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = embedded.view(1, batch_size, self.embedding_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, hidden)
        
        # Calculate attention from current RNN hidden state and all encoder outputs;
        context, attn_weights = self.attn(rnn_output, encoder_outputs)

        rnn_output = rnn_output.squeeze(0) # S=1 x B x N --> B x N
        context = context.squeeze(1) # B x S=1 x N --> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally, predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        return output, attn_weights, hidden
        






        













        
