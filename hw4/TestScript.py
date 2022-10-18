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


ipt = torch.tensor((10,), dtype=int)
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


AM = AlignmentModel(input_size, output_size, hidden_size, hidden_align_size)

c_i = AM.forward(ipt, trg, 1)


# print(c_i[0].size())
