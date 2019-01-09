from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pickle
import time
import gc

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import pdb
import sacrebleu

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from model_architectures import Encoder_RNN, Decoder_RNN
from data_prep import prepareTrainData, tensorsFromPair, prepareNonTrainDataForLanguagePair, load_cpickle_gc
from inference import generate_translation
from misc import timeSince, load_cpickle_gc

device = "cpu"


BATCH_SIZE = 32
PAD_token = 0
PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
teacher_forcing_ratio = 1.0
attn_model = 'dot'


class LanguagePairDataset(Dataset):
    
    def __init__(self, sent_pairs): 
        # this is a list of sentences 
        self.sent_pairs_list = sent_pairs

    def __len__(self):
        return len(self.sent_pairs_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        sent1 = self.sent_pairs_list[key][0]
        sent2 = self.sent_pairs_list[key][1]
        return [sent1, sent2, len(sent1), len(sent2)]

def language_pair_dataset_collate_function(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    sent1_list = []
    sent1_length_list = []
    sent2_list = []
    sent2_length_list = []
    # padding
    # NOW PAD WITH THE MAXIMUM LENGTH OF THE FIRST and second batches 
    max_length_1 = max([len(x[0]) for x in batch])
    max_length_2 = max([len(x[1]) for x in batch])
    for datum in batch:
        padded_vec_1 = np.pad(np.array(datum[0]).T.squeeze(), pad_width=((0,max_length_1-len(datum[0]))), 
                                mode="constant", constant_values=PAD_token)
        padded_vec_2 = np.pad(np.array(datum[1]).T.squeeze(), pad_width=((0,max_length_2-len(datum[1]))), 
                                mode="constant", constant_values=PAD_token)
        sent1_list.append(padded_vec_1)
        sent2_list.append(padded_vec_2)
        sent1_length_list.append(len(datum[0]))
        sent2_length_list.append(len(datum[1]))
    return [torch.from_numpy(np.array(sent1_list)), torch.LongTensor(sent1_length_list), 
            torch.from_numpy(np.array(sent2_list)), torch.LongTensor(sent2_length_list)]


#input_lang, target_lang, train_pairs = prepareTrainData(
#    "iwslt-vi-en-processed/train.tok.vi",
#    "iwslt-vi-en-processed/train.tok.en",
#    input_lang = 'vi',
#    target_lang = 'en')

_, _, test_pairs= prepareTrainData(
    "iwslt-vi-en-processed/test.vi",
    "iwslt-vi-en-processed/test.en",
    input_lang = 'vi',
    target_lang = 'en')

input_lang = load_cpickle_gc("input_lang_vi")
target_lang = load_cpickle_gc("target_lang_en")

test_idx_pairs = []
for x in test_pairs:
    indexed = list(tensorsFromPair(x, input_lang, target_lang))
    test_idx_pairs.append(indexed)

train_idx_pairs = load_cpickle_gc("train_vi_en_idx_pairs")
train_idx_pairs = train_idx_pairs[:-5]
val_idx_pairs = load_cpickle_gc("val_idx_pairs")
val_pairs = load_cpickle_gc("val_pairs")

print(len(train_idx_pairs))


#train_idx_pairs = []
#for x in train_pairs:
#    indexed = list(tensorsFromPair(x, input_lang, target_lang))
#    train_idx_pairs.append(indexed)

#pickle.dump(input_lang, open("input_lang_vi", "wb"))
#pickle.dump(target_lang, open("target_lang_en", "wb"))
#pickle.dump(train_idx_pairs, open("train_vi_en_idx_pairs", "wb"))

#_, _, val_pairs = prepareTrainData("iwslt-vi-en-processed/dev.vi","iwslt-vi-en-processed/dev.en",input_lang = 'vi',target_lang = 'en')

#val_idx_pairs = []
#for x in val_pairs:
#    indexed = list(tensorsFromPair(x, input_lang, target_lang))
#    val_idx_pairs.append(indexed)

#pickle.dump(val_pairs, open("val_pairs", "wb"))
#pickle.dump(val_idx_pairs, open("val_idx_pairs", "wb"))




train_dataset = LanguagePairDataset(train_idx_pairs)
# is there anything in the train_idx_pairs that is only 0s right noww instea dof padding. 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=language_pair_dataset_collate_function,
										   shuffle=True
                                          )

val_dataset = LanguagePairDataset(val_idx_pairs[:500])
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=1, 
                                           collate_fn=language_pair_dataset_collate_function,
                                          )

test_dataset = LanguagePairDataset(test_idx_pairs)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=1, 
                                           collate_fn=language_pair_dataset_collate_function,
                                          )

class Encoder_Batch_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_Batch_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, sents, sent_lengths):
        '''
            sents is (batch_size by padded_length)
            when we evaluate sentence by sentence, you evaluate it with batch_size = 1, padded_length.
            [[1, 2, 3, 4]] etc. 
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = torch.tensor(descending_lengths)
        descending_indices = torch.tensor(descending_indices).to(device)
        descending_sents = torch.index_select(sents, torch.tensor(0), descending_indices)
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # rnn_out is 32 by 72 by 256
        
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))  
        rnn_out = torch.index_select(rnn_out, 0, torch.LongTensor(change_it_back).to(device)) 
        
        return rnn_out, self.hidden
    

class Encoder_Batch_Bidir_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_Batch_Bidir_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        
    def init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size, device=device)

    def forward(self, sents, sent_lengths):
        '''
            sents is (batch_size by padded_length)
            when we evaluate sentence by sentence, you evaluate it with batch_size = 1, padded_length.
            [[1, 2, 3, 4]] etc. 
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = torch.tensor(descending_lengths)
        descending_indices = torch.tensor(descending_indices).to(device)
        descending_sents = torch.index_select(sents, torch.tensor(0), descending_indices)
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # rnn_out is 32 by 72 by 256
        
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))  
        rnn_out = torch.index_select(rnn_out, 0, torch.LongTensor(change_it_back).to(device)) 
 
        # self.hidden is 4 by 8 by 256
        # let's only use the top-most layer for the encoder output
        # so we want to return 8 by 512
        hidden_top = torch.cat((self.hidden[2], self.hidden[3]), dim=1)
        hidden_bottom = torch.cat((self.hidden[0], self.hidden[1]), dim=1)
        self.hidden = torch.stack((hidden_top, hidden_bottom))
        
        return rnn_out, self.hidden

    
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.softmax = torch.nn.Softmax(dim=1)
        
        if self.method == 'general':
            self.attn = nn.Linear(2*self.hidden_size, 2*hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs, attn_mask):

        # Create variable to store attention energies
        # hidden is 16 by 512
        # encoder_outputs is 16 by 72 by 512
        
        # this just uses the top layer of the 2-layer decoder. 
        # okay?
        hidden = hidden.squeeze(0)
        batch_size = hidden.size()[0]
        attn_energies = []
        for i in range(batch_size):
            attn_energies.append(self.score(hidden[i], encoder_outputs[i]))
        
        attn_energies = torch.stack(attn_energies).squeeze(0)
        # attn_energies is 32 by 72
        if attn_mask is not None:
            attn_energies = attn_mask * attn_energies
            attn_energies[attn_energies == 0] = -1e10
        # i want to mask the attention energies
        if attn_mask is None:
            attn_energies = attn_energies.view(1, -1)
        attn_energies = self.softmax(attn_energies)
        
        context_vectors = []
        for i in range(batch_size):
            context_vectors.append(torch.matmul(attn_energies[i], encoder_outputs[i]))
                
        context_vectors = torch.stack(context_vectors)
        
        return context_vectors
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':            
            # hidden is 1 by 256
            # encoder_output is 22 by 256
            encoder_output = torch.transpose(encoder_output, 0, 1)
            # encoder_output is 256 by 22
            energy = torch.matmul(hidden, encoder_output)
            return energy
        
        elif self.method == 'general':
            # hidden is 1 by 256
            # encoder_output is 256 by 22
            # encoder_output = torch.transpose(encoder_output, 0, 1)
            hidden = hidden.view(1, -1)
            transformed = self.attn(encoder_output)
            transformed = torch.transpose(transformed, 0, 1)
            energy = torch.matmul(hidden, transformed)
            return energy[0]
        
        elif self.method == 'concat':
            len_encoder_output = encoder_output.size()[1]
            # hidden is 1 by 256
            # encoder_output is 256 by 22
            hidden = torch.transpose(hidden, 0, 1)
            # hidden is 256 by 1
            hidden = hidden.repeat(hidden_size, len_encoder_output)
            # hidden is 256 by 22
            concat = torch.cat((hidden, encoder_output), dim=0)
            # concat is 512 by 22
            # self.attn(concat) --> 256 by 22
            energy = torch.matmul(self.v, F.tanh(self.attn(concat)))
            return energy
        
        
        
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=2):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define layers
        self.embedding = nn.Embedding(output_size, 2*hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(2*hidden_size, 2*hidden_size, num_layers = n_layers)
        self.concat = nn.Linear(hidden_size * 4, hidden_size*2)
        self.out = nn.Linear(hidden_size*2, output_size)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, attn_mask):
        # Note: we run this one step at a time

        # input_seq: 16 by 1
        # last_hidden: 2 by 16 by 512 
        # encoder_outputs: 16 by 57 by 512 
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        #if batch_size == 1:
        #    pdb.set_trace()
        embedded = self.embedding(input_seq)
        embedded = embedded.view(1, batch_size, -1)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        context = self.attn(rnn_output, encoder_outputs, attn_mask)
        context = context.view(batch_size, 2*hidden_size)
        # context is 32 by 256

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.view(batch_size, 2*hidden_size) # S=1 x B x N -> B x N
        # rnn_output is 32 by 256
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # output is 32 by vocab_size
        output = self.LogSoftmax(output)

        # Return final output, hidden state
        return output, hidden
    
    
def calculate_bleu(predictions, labels):
    """
    Only pass a list of strings 
    """
    # tthis is ony with n_gram = 4

    bleu = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
    return bleu


def beam_search(decoder, decoder_input, encoder_outputs, hidden, max_length, k, target_lang):
    
    candidates = [(decoder_input, 0, hidden)]
    potential_candidates = []
    completed_translations = []

    # put a cap on the length of generated sentences
    for m in range(max_length):
        for c in candidates:
            # unpack the tuple
            c_sequence = c[0]
            c_score = c[1]
            c_hidden = c[2]
            # EOS token
            if c_sequence[-1] == EOS_token:
                completed_translations.append((c_sequence, c_score))
                k = k - 1
            else:
                # pdb.set_trace()
                next_word_probs, hidden = decoder(torch.cuda.LongTensor([c_sequence[-1]]).view(1, 1), torch.cuda.FloatTensor(c_hidden), encoder_outputs, attn_mask = None)
                next_word_probs = next_word_probs[0]
                # in the worst-case, one sequence will have the highest k probabilities
                # so to save computation, only grab the k highest_probability from each candidate sequence
                top_probs, top_idx = torch.topk(next_word_probs, k)
                for i in range(len(top_probs)):
                    word = top_idx[i].reshape(1, 1).to(device)
                    new_score = c_score + top_probs[i]
                    potential_candidates.append((torch.cat((c_sequence, word)).to(device), new_score, hidden))

        candidates = sorted(potential_candidates, key= lambda x: x[1], reverse=True)[0:k] 
        potential_candidates = []

    completed = completed_translations + candidates
    completed = sorted(completed, key= lambda x: x[1], reverse=True)[0] 
    final_translation = []
    for x in completed[0]:
        final_translation.append(target_lang.index2word[x.squeeze().item()])
    return final_translation


def generate_translation(encoder, decoder, sentence, max_length, target_lang, search="greedy", k = None):
    """ 
    @param max_length: the max # of words that the decoder can return
    @returns decoded_words: a list of words in target language
    """    
    with torch.no_grad():
        input_tensor = sentence
        input_length = sentence.size()[1]
        
        # encode the source sentence
        encoder_hidden = encoder.init_hidden(1)
        # input_tensor 1 by 12 
        # 
        encoder_outputs, encoder_hidden = encoder(input_tensor.view(1, -1),torch.tensor([input_length]))
        # start decoding
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        
        if search == 'greedy':
            decoded_words = greedy_search_batch(decoder, decoder_input, encoder_outputs, decoder_hidden, max_length)
        elif search == 'beam':
            if k == None:
                k = 2 # since k = 2 preforms badly
            decoded_words = beam_search(decoder, decoder_input, encoder_outputs, decoder_hidden, max_length, k, target_lang) 

        return decoded_words
    
    
def test_model(encoder, decoder, search, test_idx_pairs, lang2, max_length, which = None):
    # for test, you only need the lang1 words to be tokenized,
    # lang2 words is the true labels
    encoder.eval()
    decoder.eval()
    
    translated_predictions = []
    if which == "test":
        loader = test_loader
        true_labels = [pair[1] for pair in test_pairs[:len(test_idx_pairs)]]
    else:
        loader = val_loader
        true_labels = [pair[1] for pair in val_pairs[:len(val_loader)]]

    for step, (sent1, sent1_length, sent2, sent2_length) in enumerate(loader):
        sent1, sent2 = sent1.to(device), sent2.to(device) 
        sent1_length, sent2_length = sent1_length.to(device), sent2_length.to(device)
        
        decoded_words = generate_translation(encoder, decoder, sent1, max_length, lang2, search=search)
        translated_predictions.append(" ".join(decoded_words).replace(" &apos;", "'").replace('SOS ', '').replace('EOS', '').replace('. ', '.').replace(' ,', ','))

    rands = random.sample(range(0, 100), 5)
    for r in rands:
        print(translated_predictions[r])
        print(true_labels[r])
    bleurg = calculate_bleu(translated_predictions, true_labels)
    return bleurg

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
            print(obj)
            break

def sequence_mask(sequence_length, device = 'cuda'):
    max_len = sequence_length.max().item()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size,1])
    seq_range_expand = seq_range_expand.to(device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return (seq_range_expand < seq_length_expand).float()


def train(sent1_batch, sent1_length_batch, sent2_batch, sent2_length_batch, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
        
    batch_size = sent1_batch.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    encoder_outputs, encoder_hidden = encoder(sent1_batch, sent1_length_batch)
    # the below code was used for a 1-layer bidirectional GRU
    # encoder_hidden = torch.cat((encoder_hidden[0, :, :], encoder_hidden[1, :, :]), 1)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([SOS_token] * batch_size).view(-1, 1).to(device)
    
    max_trg_len = max(sent2_length_batch)
    loss = 0
    attn_mask = sequence_mask(sent1_length_batch)
    
    # Run through decoder one time step at a time using TEACHER FORCING=1.0
    for t in range(max_trg_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, attn_mask
        )
        # decoder_output is 32 by vocab_size
        # sent2_batch is 32 by 46
        loss += criterion(decoder_output, sent2_batch[:, t])
        decoder_input = sent2_batch[:, t]
        
    
    loss = loss / max_trg_len.float()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.1)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.1)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()
    
    
def trainIters(encoder, decoder, n_epochs, validation_pairs, lang1, lang2, search, title, max_length_generation, print_every, val_every, learning_rate):
    start = time.time()
    count, print_loss_total = 0, 0
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=5*learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token) # this ignores the padded token. 

    for epoch in range(n_epochs):
        for step, (sent1s, sent1_lengths, sent2s, sent2_lengths) in enumerate(train_loader):
            encoder.train()
            decoder.train()
            sent1_batch, sent2_batch = sent1s.to(device), sent2s.to(device) 
            sent1_length_batch, sent2_length_batch = sent1_lengths.to(device), sent2_lengths.to(device)
            
            loss = train(sent1_batch, sent1_length_batch, sent2_batch, sent2_length_batch, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            
            print_loss_total += loss
            count += 1
          
            if (step+1) % print_every == 0:
                # lets train and plot at the same time. 
                print_loss_avg = print_loss_total / count
                count = 0
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, step / n_epochs),
                                                         step, step / n_epochs * 100, print_loss_avg))
                print("Memory allocated (mb): ", torch.cuda.memory_allocated(device)/(1e6))

                if (step+1) % val_every == 0:
                    with torch.no_grad():
                        bleu_score = test_model(encoder, decoder, search, validation_pairs, lang2, max_length=max_length_generation)
                    # returns bleu score
                    print("VALIDATION BLEU SCORE: "+str(bleu_score))
                    torch.save(encoder.state_dict(), "Attention_Vish_encoder_latest")
                    torch.save(decoder.state_dict(), "Attention_Vish_decoder_latest")
                    
                           
            del sent1s, sent1_lengths, sent2s, sent2_lengths, sent1_batch, sent2_batch, sent1_length_batch, sent2_length_batch
            gc.collect() 
            
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 350
encoder1 = Encoder_Batch_Bidir_RNN(input_lang.n_words, hidden_size).to(device)
decoder1 = LuongAttnDecoderRNN(attn_model, hidden_size, target_lang.n_words).to(device)
encoder1.load_state_dict(torch.load("Attention_Vish_encoder_latest"))
# decoder1 = Decoder_Batch_2RNN(target_lang.n_words, hidden_size).to(device)
decoder1.load_state_dict(torch.load("Attention_Vish_decoder_latest"))

bleu_score = test_model(encoder1, decoder1, "beam", test_idx_pairs, target_lang, max_length=25, which="test")
print(bleu_score)


args = {
    'n_epochs': 4,
    'learning_rate': 0.000003,
    'search': 'beam',
    'encoder': encoder1,
    'decoder': decoder1,
    'lang1': input_lang, 
    'lang2': target_lang,
    "validation_pairs": val_idx_pairs, 
    "title": "Training Curve for Basic 1-Directional Encoder Decoder Model With LR = 1.2",
    "max_length_generation": 25, 
    "print_every": 100,
    "val_every": 1000
}

"""
We follow https://arxiv.org/pdf/1406.1078.pdf 
and use the Adadelta optimizer

"""
print(BATCH_SIZE)

#trainIters(**args)
