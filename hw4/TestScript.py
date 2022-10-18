import torch
import torch.nn as nn
from seq2seq import Embedder, EncoderRNN, AlignmentHelper, AlignmentModel, AttnDecoderRNN

device = 'cpu'
MAX_LENGTH = 15


input_size = 620
hidden_size = 1000
hidden_align_size = hidden_size
output_size = 650

E = EncoderRNN(input_size, hidden_size)
AH = AlignmentHelper(hidden_size, hidden_align_size)


ipt = torch.tensor((10,12, 13, 14, 15), dtype=int)
trg = torch.tensor((15,), dtype=int)

ipt = ipt.unsqueeze(-1)
trg = trg.unsqueeze(-1)

h0 = E.get_initial_hidden_state()
s0 = h0

z_i_forward, h_i_forward = E.forward(ipt, h0)


print("encoder forward output:\n", z_i_forward.shape, h_i_forward.shape)

z_i_backward, h_i_backward = E.backward(ipt, h0)
print("encoder backward output\n", z_i_backward.shape, h_i_backward.shape)

e_ij = AH.forward(s0, h_i_forward, h_i_backward)
print("e_ij:\n", e_ij)


# AM = AlignmentModel(input_size, output_size, hidden_size, hidden_align_size)

# c_i = AM.forward(ipt, trg, 1)


# print(c_i[0].size())


#%%


##### OLD #####
# class AlignmentModel(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, hidden_align_size):
#         super().__init__()
        
#         self.hidden_size = hidden_size
#         self.ah = AlignmentHelper(hidden_size, hidden_align_size)
#         self.encoder = EncoderRNN(input_size, hidden_size)
#         self.decoder = AttnDecoderRNN(input_size, hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, src_sent, y, cnt):
#         # src_sent_flipped = src_sent.T
#         src_sent_flipped = torch.flip(src_sent, dims=[0,1])
#         h_i_forward = []
#         h_i_backward = []
        
#         s_i = []
#         e_vals = []
#         alphas = []
        
#         context_vecs = []
        
#         for i in range(cnt):
#             if len(h_i_forward) == 0:    
#                 _, enc_forward_out = self.encoder.forward(src_sent[i], self.encoder.get_initial_hidden_state())
#                 _, enc_backward_out = self.encoder.backward(src_sent_flipped[i], self.encoder.get_initial_hidden_state())
                
#                 h_i_forward.append(enc_forward_out)
#                 h_i_backward.append(enc_backward_out)
#                 # print("forward dim\n", h_i_forward[i].size())
#                 # print(f'{h_i_backward[i].size()=}')
                
#                 # print("s_i", s_i[i].size())
#                 e_vals.append(self.ah.forward(self.decoder.get_initial_hidden_state(), h_i_forward[i], h_i_backward[i]))
#                 # print("e_vals:\n", e_vals[i].size())
#                 alphas.append(self.softmax(e_vals[i]))
                
#                 # print("alphas shape\n", alphas[i].size())
                
#                 h_new = torch.cat((h_i_forward[i], h_i_backward[i]), dim=1)
                
#                 # print("h_new", h_new.size())
#                 context_vecs.append(alphas[i] @ torch.cat((h_i_forward[i], h_i_backward[i]), dim=-1))
#                 dec_out = self.decoder.forward(y, self.decoder.get_initial_hidden_state(), context_vecs[i])
                
                
#             else:
#                 _, enc_forward_out = self.encoder.forward(src_sent[i], h_i_forward[i-1])
#                 _, enc_backward_out = self.encoder.backward(src_sent_flipped[i], h_i_backward[i-1])
#                 h_i_forward.append(enc_forward_out)
#                 h_i_backward.append(enc_backward_out)
#                 s_i.append(dec_out)
#                 e_vals.append(self.ah.forward(s_i[i-1], h_i_forward[i], h_i_backward[i]))
#                 alphas.append(self.softmax(e_vals[:]))
#                 context_vecs.append(alphas[i] @ torch.cat((h_i_forward[i], h_i_backward[i]), dim=-1))
#                 dec_out = self.decoder.forward(y, s_i[i-1], context_vecs[i])
#                 s_i.append(dec_out)
                
        
#         return context_vecs