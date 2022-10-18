import torch
import torch.nn as nn

device = 'cpu'
MAX_LENGTH = 15

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

class AlignmentHelper(nn.Module):
    def __init__(self, hidden_size, hidden_align_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_align_size)
        self.Va = nn.Linear(1, hidden_align_size)
        self.Ua = nn.Linear(2*hidden_size, hidden_align_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, s, h_forward, h_backward):
        h_c = torch.cat((h_forward, h_backward), dim=-1) # get bidirection annotations
        
        
        step1 = self.tanh(self.Wa(s) + self.Ua(h_c)).squeeze(0).T
        step2 = self.Va.weight.T
        
        
        
        e_ij = step2 @ step1
        
        return e_ij


class AlignmentModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_align_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.ah = AlignmentHelper(hidden_size, hidden_align_size)
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(input_size, hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, src_sent, y, cnt):
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
                
                h_i_forward.append(enc_forward_out)
                h_i_backward.append(enc_backward_out)
                # print("forward dim\n", h_i_forward[i].size())
                # print(f'{h_i_backward[i].size()=}')
                
                # print("s_i", s_i[i].size())
                e_vals.append(self.ah.forward(self.decoder.get_initial_hidden_state(), h_i_forward[i], h_i_backward[i]))
                # print("e_vals:\n", e_vals[i].size())
                alphas.append(self.softmax(e_vals[i]))
                
                # print("alphas shape\n", alphas[i].size())
                
                h_new = torch.cat((h_i_forward[i], h_i_backward[i]), dim=1)
                
                # print("h_new", h_new.size())
                context_vecs.append(alphas[i] @ torch.cat((h_i_forward[i], h_i_backward[i]), dim=-1))
                dec_out = self.decoder.forward(y, self.decoder.get_initial_hidden_state(), context_vecs[i])
                
                
            else:
                _, enc_forward_out = self.encoder.forward(src_sent[i], h_i_forward[i-1])
                _, enc_backward_out = self.encoder.backward(src_sent_flipped[i], h_i_backward[i-1])
                h_i_forward.append(enc_forward_out)
                h_i_backward.append(enc_backward_out)
                s_i.append(dec_out)
                e_vals.append(self.ah.forward(s_i[i-1], h_i_forward[i], h_i_backward[i]))
                alphas.append(self.softmax(e_vals[:]))
                context_vecs.append(alphas[i] @ torch.cat((h_i_forward[i], h_i_backward[i]), dim=-1))
                dec_out = self.decoder.forward(y, s_i[i-1], context_vecs[i])
                s_i.append(dec_out)
                
        
        return context_vecs
    
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
        
        # make sure encoder output is correctly used for c_i.
        
        r_i = self.sigmoid(self.Wr(y) + self.Ur(hidden) + self.Cr(c_i))
        z_i = self.sigmoid(self.Wz(y) + self.Uz(hidden) + self.Cz(c_i))
        s_tilde = self.tanh(self.Ws(y) + self.Us(hidden) + self.Cs(c_i))
        
        s_i = (1- z_i) * hidden + z_i * s_tilde
        return s_i
        # return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
input_size = 620
hidden_size = 1000
hidden_align_size = hidden_size
output_size = 650

E = EncoderRNN(input_size, hidden_size)
AH = AlignmentHelper(hidden_size, hidden_align_size)


ipt = torch.tensor((10,), dtype=int)
trg = torch.tensor((15,), dtype=int)

ipt = ipt.unsqueeze(-1)
trg = trg.unsqueeze(-1)




h0 = E.get_initial_hidden_state()
s0 = h0

z_i_forward, h_i_forward = E.forward(ipt, h0)
print("encoder forward output:\n", z_i_forward.shape, h_i_forward.shape)

z_i_backward, h_i_backward = E.backward(ipt, h0)
print("encoder backward output:\n", z_i_backward.shape, h_i_backward.shape)

e_ij = AH.forward(s0, h_i_forward, h_i_backward)
print("e_ij:\n", e_ij)


AM = AlignmentModel(input_size, output_size, hidden_size, hidden_align_size)

c_i = AM.forward(ipt, trg, 1)


# print(c_i[0].size())
