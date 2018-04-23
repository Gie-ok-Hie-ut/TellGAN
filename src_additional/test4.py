import os

import torch
import torch.nn as nn

import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchvision.utils

from convlstm import ConvLSTM
#from utils import *

b_size = 128

time_steps = 1
time_end = 10
batch_size = 1
input_size = 5
# THis is the # of channels returned in the output
# Should the output be only features, features+emb, *3-channel image*??????
# I'm thinking a 3 channel image, but Idk... going with features+emb for now
hidden_size = [16, 16] # = 16 + 7
num_layers = 2
emb_size=5

feat_dim_h=7
feat_dim_w=7
feat_dim_chan=16

def make_input(emb_in):
    feature = Variable(torch.randn(batch_size, feat_dim_chan, feat_dim_h, feat_dim_w))
    print("feature: ", feature.size())

    # Create shape to concat for LSTM input
    # Tile vector along feature channels, copy for each batch
    embedding = emb_in.expand(batch_size, emb_in.size(1),feat_dim_h,feat_dim_w)
    print("embedding: ", embedding.size())

    input = torch.cat((feature, embedding), dim=1)
    print("Input: ", input.size())

    # target = Variable(torch.LongTensor(batch_size).random_(0, hidden_size-1))
    return input



word_to_ix = {"hello": 0, "world": 1}
#embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
embeds = nn.Embedding(2, feat_dim_h)  # 2 words in vocab, 7 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(autograd.Variable(lookup_tensor))
lookup_tensor = torch.LongTensor([word_to_ix["world"]])
world_embed = embeds(autograd.Variable(lookup_tensor))

print("Hello Emb:", hello_embed)

encoder = ConvLSTM(input_size=(feat_dim_h,feat_dim_w),
                   input_dim=feat_dim_chan+feat_dim_h,
                   hidden_dim=hidden_size,
                   kernel_size=(3,3),
                   num_layers=2,
                  )

encoder.cuda()

crit = nn.MSELoss() #nn.BCELoss()
crit.cuda()

threshold = nn.Threshold(0., 0.0)

params = list(encoder.parameters())
optimizer = optim.Adam(params, lr=0.001)


s = 1
input = None
hidden_states = None
for e in range(5):
    optimizer.zero_grad()
    input = None
    if input is None:
        # Make first Sequence (need a (seq x batch x chan x h x w) tensor)
        input =  make_input(hello_embed).unsqueeze(0)
    else:
        input = torch.cat((input, make_input(hello_embed).unsqueeze(0)))
    print("Real Input: ", input.size())

    target = Variable(torch.randn(batch_size, feat_dim_chan, feat_dim_h, feat_dim_w))
    print("Target: ", target.size())

    ########
    #Encoder
    ########
    #hidden = encoder.get_init_states(batch_size)
    #output, hidden_states = encoder(input.cuda().clone(), hidden)


    output, hidden_states = encoder(input.cuda().clone())



    #last_state = encoder_state[-1]
    last_output = output[-1]

    print("Output: ", last_output.size())
    #print("hidden_state: ", last_state)

    #######
    #loss##
    #######
    loss = crit(last_output.clone(), target.cuda().clone())

    loss.backward()#retain_graph=True)
    optimizer.step()

    print("Model: ", encoder)
    #if hidden_states is not None:
    #    for i, (h, c) in enumerate(hidden_states):
            #hidden_states[i] = (Variable(h.data, requires_grad=True),
            #                   Variable(c.data, requires_grad=True))
            #hidden_states[i] = (h.detach_(), c.detach_())
            #h.detach_()
            #c.detach_()

    #if i % 100 == 0:
    print("Epoch: {0} | Iter: {1} |".format(e, e))
    print("Loss: {0}".format(loss.data.cpu().numpy()[0]))
    print("===========================")


