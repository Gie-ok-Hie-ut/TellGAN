import torch
import torch.nn as nn
from torch.autograd import Variable

#LSTM

class NewNet(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(NewNet,self).__init__()
        self.model = nn.LSTM(input_size,hidden_size,num_layers)
    
    def forward(self,input_seq):
        output_seq,_ = self.model(input_seq)
        #print(output_seq) # 10 1 7
        last_output = output_seq[-1]
        #print(last_output) # 1 7


        return last_output



time_steps = 10
batch_size = 1
input_size = 5
hidden_size = 7
num_layers = 2


model = NewNet(input_size,hidden_size,num_layers)


input_seq = Variable(torch.randn(time_steps, batch_size, input_size))
target = Variable(torch.LongTensor(batch_size).random_(0, hidden_size-1))

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)

for t in range(10):
    last_output= model(input_seq)
    err = loss(last_output, target)

    optimizer.zero_grad()
    err.backward()
    optimizer.step()

    #print(loss)
    print(last_output)

