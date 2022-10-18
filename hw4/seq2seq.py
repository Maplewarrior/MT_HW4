#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


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
        
        # Get weight matrices
        self.Wh_forward = nn.Linear(self.input_size, self.hidden_size) # W = n x m
        self.Wz_forward = nn.Linear(self.input_size, self.hidden_size)
        self.Wr_forward = nn.Linear(self.input_size, self.hidden_size)
        
        self.Uh_forward = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uz_forward = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ur_forward = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Get weight matrices for backward pass
        self.Wh_backward = nn.Linear(self.input_size, self.hidden_size) # W = n x m
        self.Wz_backward = nn.Linear(self.input_size, self.hidden_size)
        self.Wr_backward = nn.Linear(self.input_size, self.hidden_size)
        
        self.Uh_backward = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uz_backward = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ur_backward = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # return output, hidden


    def forward(self, x, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        x = self.embed(x)
        
        r_i = self.sigmoid(self.Wr_forward(x) + self.Ur_forward(hidden))
        z_i = self.sigmoid(self.Wz_forward(x) + self.Uz_forward(hidden)) # = output
        # compute hidden at i
        h_i = self.tanh(self.Wh_forward(x) + self.Uh_forward(r_i * hidden)) # = hidden
        
        return z_i, h_i
    
    def backward(self, x, hidden):
        """runs the backward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        x = self.embed(x)
        
        r_i = self.sigmoid(self.Wr_backward(x) + self.Ur_backward(hidden))
        z_i = self.sigmoid(self.Wz_backward(x) + self.Uz_backward(hidden)) # = output
        # compute hidden at i
        h_i = self.tanh(self.Wh_backward(x) + self.Uh_backward(r_i * hidden)) # = hidden
        
        return z_i, h_i
        # return output, hidden

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# #%%
input_size = 620
hidden_size = 1000
hidden_align_size = hidden_size
ipt = torch.ones(1, 1, dtype = int)

E = EncoderRNN(input_size, hidden_size)


out, h = E.forward(ipt, E.get_initial_hidden_state())

print(out.size())
print(h.size())

out_b, h_b =  E.backward(ipt, E.get_initial_hidden_state())



#%%
h_c = torch.cat((h, h_b), dim=-1)
Ua = nn.Linear(2*hidden_size, hidden_align_size)

# print(Ua(h_c).size())

Ws = nn.Linear(hidden_size, hidden_size)
s0 = Ws(h_b)
# print(Ws(s0).size())
# v = nn.Linear(hidden_size)

res = (Ws(s0) + Ua(h_c))

#%%


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
        e_ij = self.Va.weight.T @ self.tanh(self.Wa(s) + self.Ua(h_j))
        return e_ij
    
        

class AlignmentModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_align_size=hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.ah = AlignmentHelper(hidden_size, hidden_align_size)
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(input_size, hidden_size, output_size)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, src_sent, y, c_prev, cnt):
        # src_sent_flipped = src_sent.T
        src_sent_flipped = torch.flip(src_sent, dims=[0,1])
        h_i_forward = []
        h_i_backward = []
        
        s_i = []
        e_vals = []
        alphas = []
        
        context_vecs = []
        
        for i in range(cnt):
            if len(h_i_forward) == 0:    
                _, enc_forward_out = self.encoder.forward(src_sent[i], self.encoder.get_initial_hidden_state())
                _, enc_backward_out = self.encoder.backward(src_sent_flipped[i], self.encoder.get_initial_hidden_state())
                dec_out = self.decoder.forward(y, self.decoder.get_initial_hidden_state(), c_prev)
                h_i_forward.append(enc_forward_out)
                h_i_backward.append(enc_backward_out)
                print("forward dim\n", h_i_forward[i].size())
                print(f'{h_i_backward[i].size()=}')
                
                s_i.append(dec_out)
                print("s_i", s_i[i].size())
                e_vals.append(self.ah.forward(s_i[i], h_i_forward[i], h_i_backward[i]))
                print("e_vals:\n", e_vals[i].size())
                alphas.append(self.softmax(e_vals[i]))
                
                print("alphas shape\n", alphas[i].size())
                
                
                
                h_new = torch.cat((h_i_forward[i], h_i_backward[i]), dim=1)
                
                print("h_new", h_new.size())
                context_vecs.append(alphas[i] @ torch.cat((h_i_forward[i], h_i_backward[i]), dim=1))
                
                
            else:
                _, enc_forward_out = self.encoder.forward(src_sent[i], h_i_forward[i-1])
                _, enc_backward_out = self.encoder.backward(src_sent_flipped[i], h_i_backward[i-1])
                dec_out = self.decoder.forward(y, s_i[i-1], context_vecs[i-1])
                h_i_forward.append(enc_forward_out)
                h_i_backward.append(enc_backward_out)
                s_i.append(dec_out)
                e_vals.append(self.ah.forward(s_i[i], h_i_forward[i], h_i_backward[i]))
                alphas.append(self.softmax(e_vals[i]))
                context_vecs.append(alphas[i] @ torch.cat((h_i_forward[i], h_i_backward[i]), dim=-1))
        
        return context_vecs
                
            
            
A = AlignmentHelper(hidden_size, hidden_align_size)

alp = A.forward(s0, h, h_b)
print(alp.size())
#%%
class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.input_size = input_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropout = nn.Dropout(self.dropout)
        
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"
        # raise NotImplementedError
        self.embed = Embedder(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) ## Check dim when rest of code is set up
        
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
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        
    def forward(self, input, hidden, c_i):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        
        "*** YOUR CODE HERE ***"
        # raise NotImplementedError
        y = self.embed(input)
        print("y shape:", y.size())
        # make sure encoder output is correctly used for c_i.
        
        
        r1 = self.Wr(y) + self.Ur(hidden)
        print("r1", r1.size())
        
        r2 = self.Cr(c_i)
        print("r2", r2.size())
        r_i = self.sigmoid(self.Wr(y) + self.Ur(hidden) + self.Cr(c_i))
        z_i = self.sigmoid(self.Wz(y) + self.Uz(hidden) + self.Cz(c_i))
        s_tilde = self.tanh(self.Ws(y) + self.Us(hidden) + self.Cs(c_i))
        
        s_i = (1- z_i) * hidden + z_i * s_tilde
        return s_i
        # return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#%%
input_size = 620
output_size = 650
hidden_size = 1000
AM = AlignmentModel(input_size, output_size, hidden_size)

AM.forward(torch.ones((10,1), dtype=int), torch.ones((10,1), dtype=int), torch.zeros((hidden_size, 2*hidden_size, )), 1)
#%%

"""
implement generic RNN class that combines the two methods above and feed that to train?

"""

class biRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.1, max_length=MAX_LENGTH):
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size, dropout=dropout, max_length=MAX_LENGTH)
        


def initialize_parameters(model):
    
    print("not implemeneted yet, look at page 14 of paper to see what the weights should be initialized as")
    
    return model
######################################################################

# def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
def train(input_tensor, target_tensor, model, criterion, n_iter,lr, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_initial_hidden_state()

    # make sure the encoder and decoder are in training mode so dropout is applied
    model.train()
    loss = []
    
    params = model.params()
    
    opt = optim.SGD(params=params, lr=lr)
    "*** YOUR CODE HERE ***"
    # raise NotImplementedError
    
    
    # USE SGD!
    

    return loss.item() 



######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    
    "*** YOUR CODE HERE ***"
    raise NotImplementedError


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################
#%%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()
    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    # encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    # decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)
    
    print("SIZE:", tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))[0].size())
    
    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        print(input_tensor)
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
