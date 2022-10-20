from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################

class Embedder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        # self.input_size = input_size
        self.embed = nn.Embedding(hidden_size, input_size)
        
    def forward(self, x):
        out = self.embed(x)
        # reshape to embedding dim x src_len
        # out = out.squeeze(1)
        return out

class Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wh = nn.Linear(self.input_size, self.hidden_size) # W = n x m
        self.Wz = nn.Linear(self.input_size, self.hidden_size)
        self.Wr = nn.Linear(self.input_size, self.hidden_size)

        self.Uh = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uz = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ur = nn.Linear(self.hidden_size, self.hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden):
        r_i = self.sigmoid(self.Wr(x) + self.Ur(hidden))
        z_i = self.sigmoid(self.Wz(x) + self.Uz(hidden))        
        h_i_bar = self.tanh(self.Wr(x) + self.Uh(r_i * hidden)) # = hidden
        h_i = (1- z_i) * hidden + z_i * h_i_bar

        return h_i

class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "*** YOUR CODE HERE ***"
        # raise NotImplementedError
        self.input_size = input_size
        self.embed = Embedder(self.hidden_size, self.input_size)
        self.hidden_size = hidden_size
        
        self.Cell_f = Cell(self.input_size, self.hidden_size) # forward sentence
        self.Cell_b = Cell(self.input_size, self.hidden_size) # backward sentence
        
        
    def forward(self, x, x_flipped, hidden_f, hidden_b):
        x = self.embed(x)
        x_flipped = self.embed(x_flipped)
        h_i_f = self.Cell_f(x, hidden_f)
        h_i_b = self.Cell_b(x_flipped, hidden_b)
        
        return h_i_f, h_i_b
    
    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class AlignmentHelper(nn.Module):
    def __init__(self, hidden_size, hidden_align_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_align_size)
        self.Va = nn.Linear(hidden_align_size, 1)
        self.Ua = nn.Linear(2*hidden_size, hidden_align_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, s, h_forward, h_backward):
        
        h_j = torch.cat((h_forward, h_backward), dim=-1) # get bidirection annotations
        
        step1 = (self.tanh(self.Wa(s) + self.Ua(h_j))).squeeze(0).T
        step1 = step1.squeeze(1)
       

        e_ij = self.Va.weight @ step1
        
        return e_ij
    



class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, input_size, hidden_size, output_size, maxout=500, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.max_length = max_length
        self.maxout = maxout
        self.dropout = nn.Dropout(dropout)
        
        
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"
        # raise NotImplementedError
        self.embed = Embedder(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) 
        
        #### make sure W's and U's should be different from the ones in the encoder ####
        self.Ws = nn.Linear(self.output_size, self.hidden_size) # W = n x m
        self.Wz = nn.Linear(self.output_size, self.hidden_size)
        self.Wr = nn.Linear(self.output_size, self.hidden_size)
        
        self.Us = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uz = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ur = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.Cs = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.Cz = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.Cr = nn.Linear(2*self.hidden_size, self.hidden_size)
        
        
        self.Wo = nn.Linear(self.maxout, self.output_size)
        self.Uo = nn.Linear(self.hidden_size, 2*self.maxout)
        self.Co = nn.Linear(2*self.hidden_size, 2*self.maxout)
        self.Vo = nn.Linear(self.output_size, 2*self.maxout)
        
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        
        self.AlignHelper = AlignmentHelper(hidden_size, hidden_align_size) 
        
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        
        
        
    def forward(self, input, hidden, h_i_forwards, h_i_backwards, e_vals=[], c_idx=0):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        
        "*** YOUR CODE HERE ***"
        
        # raise NotImplementedError
        y = self.embed(input)
        y = self.dropout(y)
        
        
        encoder_hiddens = torch.zeros(2, self.max_length, 1, self.hidden_size)
        encoder_hiddens[0] = h_i_forwards
        encoder_hiddens[1] = h_i_backwards
        
        
        e_ij = self.AlignHelper.forward(hidden, encoder_hiddens[0][c_idx], encoder_hiddens[1][c_idx])
        
        
        encoder_out = torch.cat((h_i_forwards, h_i_backwards), dim=-1)
        if len(e_vals) == 0:
            # softmax of a single value is 1 :)
            c_i = 1 * encoder_out
            attention = None
            e_vals = [e_ij]
        
        else:
            e_vals.append(e_ij[0])
            e_t = torch.tensor(e_vals)
            attention = self.softmax(e_t)
            c_i = attention[-1] * encoder_out
        
        # r1 = self.Wr(y) + self.Ur(hidden)
        # r2 = self.Cr(c_i)
        
        # print(self.Cr(c_i))
        
        
        r_i = self.sigmoid(self.Wr(y) + self.Ur(hidden) ) #+ self.Cr(c_i))
        z_i = self.sigmoid(self.Wz(y) + self.Uz(hidden) + self.Cz(c_i))
        s_tilde = self.tanh(self.Ws(y) + self.Us(r_i * hidden) + self.Cs(c_i))
        s_i = (1- z_i) * hidden + z_i * s_tilde
        
        
        
        t_tilde = self.Uo(hidden) + self.Vo(y) + self.Co(c_i)
        
        
        t_i = torch.zeros(self.max_length, self.maxout)
        for i in range(self.max_length):
            for j in range(self.maxout):
                # print(t_tilde[i][0][2*j-1])
                val = max(t_tilde[i][0][2*j-1], t_tilde[i][0][2*j])
                t_i[i][j] = val
        
        print("yyy", y.size())
        print("Wooo", self.Wo.weight.size())
        print("t_iii")
        output = torch.exp(y @ self.Wo.weight @ t_i.T)
        
        
        return output, s_i, attention, e_vals
        # return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, n_iter,lr, max_length=MAX_LENGTH):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    
    encoder_hidden_f = encoder.get_initial_hidden_state()
    encoder_hidden_b = encoder.get_initial_hidden_state()
    
    decoder_hidden = decoder.get_initial_hidden_state()
    
    loss = 0

    encoder_hiddens_f = torch.zeros(len(input_tensor), 1)
    encoder_hiddens_b = torch.zeros(len(input_tensor), 1)
    
    input_tensor_flipped = torch.flip(input_tensor, dims = [0,1])
    # Loop over source sentence
    for ei in range(len(input_tensor)):
        encoder_hidden_f, encoder_hidden_b = encoder(input_tensor[ei], input_tensor_flipped[ei], encoder_hidden_f, encoder_hidden_b)
        # append hidden forward and backward
        encoder_hiddens_f[ei] = encoder_hidden_f
        encoder_hiddens_b[ei] = encoder_hidden_b
    
    e_vals = []
    for di in range(len(target_tensor)):
        d_out, decoder_hidden, attention, e_vals = decoder(target_tensor[di], decoder_hidden, encoder_hiddens_f, encoder_hiddens_b, c_idx=di)
        loss += criterion(d_out, target_tensor[di])
        if target_tensor[di] == EOS_token:
            break
    
    
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    
    return loss.item()/len(target_tensor)
    

def train()


input_size = 620
output_size = 650
hidden_size = 1000
hidden_align_size = hidden_size

ipt = torch.ones(15, 1, dtype=int)

trg = torch.ones(15, 1, dtype=int)

E = EncoderRNN(input_size, hidden_size)
AH = AlignmentHelper(hidden_size, hidden_align_size)
D = AttnDecoderRNN(input_size, hidden_size, output_size)

# ipt_flipped = torch.flip(ipt, dims=[1,0])

h_i_f, h_i_b = E.forward(ipt, ipt, E.get_initial_hidden_state(), E.get_initial_hidden_state())


out, s_i, attention, e_vals = D.forward(trg[3], D.get_initial_hidden_state(), h_i_f, h_i_b)


# print(out)

