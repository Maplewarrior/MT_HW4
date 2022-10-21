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
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # self.embed = nn.Embedding(hidden_size, input_size)
        
        if input_size > hidden_size:
            self.embed = nn.Embedding(input_size+1, input_size)
        else:
            self.embed = nn.Embedding(hidden_size, input_size)
        
    def forward(self, x):

        out = self.embed(x)
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
        self.hidden_size = hidden_size
        self.embed = Embedder(self.input_size, self.hidden_size)
        
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
        step1 = (self.tanh(self.Wa(s) + self.Ua(h_j))).squeeze(0)
        e_ij = self.Va(step1)

        return e_ij            
class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, hidden_align_size, output_size, maxout=500, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.maxout = maxout
        self.dropout = nn.Dropout(dropout)
        
        
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"
        # raise NotImplementedError
        self.embed = Embedder(output_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1) 
        
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
        
        print("h_i", h_i_forwards.size())
        encoder_hiddens = torch.zeros(2, self.max_length, 1, self.hidden_size)
        print("forward", h_i_forwards.size())
        print("backward", h_i_backwards.size())
        
        encoder_hiddens[0] = h_i_forwards
        encoder_hiddens[1] = h_i_backwards
        
        e_ij = self.AlignHelper.forward(hidden, encoder_hiddens[0][c_idx], encoder_hiddens[1][c_idx])
        print("E_IJ", e_ij.size())
        encoder_out = torch.cat((h_i_forwards, h_i_backwards), dim=-1)
        if len(e_vals) == 0:
            # softmax of a single value is 1 :)
            c_i = 1 * encoder_out
            attention = torch.zeros(15,1,1)
            val = e_ij[c_idx].item()
            e_vals = [val]
        else:

            val = e_ij[c_idx].item()
            e_vals.append(val)
            e_t = torch.tensor(tuple(e_vals))
            attention = self.softmax(e_t)
            c_i = attention[-1] * encoder_out
        
        r_i = self.sigmoid(self.Wr(y) + self.Ur(hidden) ) #+ self.Cr(c_i))
        z_i = self.sigmoid(self.Wz(y) + self.Uz(hidden) + self.Cz(c_i))
        s_tilde = self.tanh(self.Ws(y) + self.Us(r_i * hidden) + self.Cs(c_i))
        s_i = (1- z_i) * hidden + z_i * s_tilde

        t_tilde = self.Uo(hidden) + self.Vo(y) + self.Co(c_i)
        
        t_i = torch.zeros(1, self.maxout)
    
        for j in range(self.maxout):
            val = max(t_tilde[c_idx][0][2*j-1], t_tilde[c_idx][0][2*j])
            t_i[0][j] = val
        
        output = torch.exp(y.T * self.Wo.weight @ t_i.T).T
        output = F.log_softmax(output, dim=1)
        
        
        return output, s_i, attention, e_vals
        # return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(15, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):


    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    optimizer.zero_grad()
    
    encoder_hidden_f = encoder.get_initial_hidden_state()
    encoder_hidden_b = encoder.get_initial_hidden_state()
    
    decoder_hidden = decoder.get_initial_hidden_state()
    
    loss = 0
    
    encoder_hiddens_f = torch.zeros(max_length, 1, encoder_hidden_f.size(-1))
    encoder_hiddens_b = torch.zeros(max_length, 1, encoder_hidden_f.size(-1))
    
    input_tensor_flipped = torch.flip(input_tensor, dims = [0,1])

    # Loop over source sentence
    for ei in range(len(input_tensor)):

        encoder_hidden_f, encoder_hidden_b = encoder(input_tensor[ei], input_tensor[ei], encoder_hidden_f, encoder_hidden_b)
        # append hidden forward and backward input_tensor_flipped[ei]

        encoder_hiddens_f[ei] = encoder_hidden_f
        encoder_hiddens_b[ei] = encoder_hidden_b
    
    e_vals = []
    

    for di in range(len(target_tensor)):
        d_out, decoder_hidden, attention, e_vals = decoder(target_tensor[di], decoder_hidden, encoder_hiddens_f, encoder_hiddens_b,e_vals, c_idx=di)
        loss += criterion(d_out, target_tensor[di])
        if target_tensor[di] == EOS_token:
            break
    
    
    loss.backward()
    optimizer.step()
    

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
        encoder_hidden_f = encoder.get_initial_hidden_state()
        encoder_hidden_b = encoder.get_initial_hidden_state()

        encoder_outputs_all_f = torch.zeros(max_length, 1, encoder.hidden_size, device=device)
        encoder_outputs_all_b = torch.zeros(max_length, 1, encoder.hidden_size, device=device)
        
        
        input_flipped = torch.flip(input_tensor, dims=[0,1])
        for ei in range(input_length):
            encoder_hidden_f, encoder_hidden_b = encoder(input_tensor[ei],input_flipped[ei],
                                                     encoder_hidden_f,
                                                     encoder_hidden_b)
            encoder_outputs_all_f[ei] = encoder_hidden_f
            encoder_outputs_all_f[ei] = encoder_hidden_b
            # encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = decoder.get_initial_hidden_state()

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        e_vals = []
        
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention, e_vals = decoder(
                decoder_input, decoder_hidden, encoder_outputs_all_f,
                encoder_outputs_all_b, e_vals, di)
            print(decoder_attention)
            decoder_attentions[di] = decoder_attention[:,0,0]
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=1, type=int,
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

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size)#.to(device)
    decoder = AttnDecoderRNN(args.hidden_size, args.hidden_size, tgt_vocab.n_words, dropout=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)
    
    
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