import torch
import torch.nn as nn
from torch.autograd import Variable

#LSTM

class NormalLSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(NormalLSTM,self).__init__()


        self.model1 = nn.Conv1d(input_size, input_size, kernel_size = 3, stride=2, padding= 3)
        self.model2 = nn.LSTM(input_size,hidden_size,num_layers)
    
    def forward(self,input_seq):
        print(input_seq)
        x = self.model1(input_seq)
        print(x)
        output_seq,_ = self.model2(x)
        #print(output_seq) # 10 1 7
        last_output = output_seq[-1]
        print(last_output) # 1 7

        return last_output



m = nn.Conv1d(16, 33, 3, stride=2)
input1 = torch.randn(20, 16, 50)
output = m(input1)
print(input1)
print(output)

time_steps = 10
batch_size = 1
input_size = 16
hidden_size = 8
num_layers = 6

model = NormalLSTM(input_size,hidden_size,num_layers)

for t in range(1,time_steps):
    landmark = Variable(torch.zeros(t, batch_size, input_size))
    word = Variable(torch.ones(t, batch_size, input_size))

    #input_stack = torch.cat((landmark,word),1)    

    last_output= model(landmark)
    #print(last_output)

