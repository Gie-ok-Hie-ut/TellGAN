import torch
import torch.nn as nn
from torch.autograd import Variable

#LSTM

class NextFeaturesForWord(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(NextFeaturesForWord, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_seq = [
            nn.Linear(input_size,input_size)
            #nn.ReLU(True),
            #nn.Sigmoid()
        ]

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.in_layer = nn.Sequential(*self.input_seq)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda()),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda()))

    def forward(self, input):
        lstm_in = self.in_layer(input)
        x = self.init_hidden()

        #Y= torch.zeros(self.num_layers, 1, self.hidden_size)
        #state = Variable((Y,Y))

        
        pred_seq, hidden = self.lstm(lstm_in,self.init_hidden())
        out = pred_seq[-1]
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        self.fc_in = nn.Linear(input_size, hidden_size)


        self.fc_out = [
            nn.Linear(hidden_size*2,hidden_size)]
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, 1, self.hidden_size)) # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, 1, self.hidden_size))
        
        # Input
        
        x1 = self.fc_in(x)
        print(x1)
        # Forward propagate LSTM
        out, _ = self.lstm(x1, (h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        print(out[-1])

        # Decode the hidden state of the last time step
        out = self.fc_out(out[-1])
        return out

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



#m = nn.Conv1d(16, 33, 3, stride=2)
#input1 = torch.randn(20, 16, 50)
#output = m(input1)
#print(input1)
#print(output)

time_steps = 10
batch_size = 1
input_size = 16
hidden_size = 8
num_layers = 6

model = BiLSTM(input_size,hidden_size,num_layers)

for t in range(1,time_steps):
    landmark = Variable(torch.zeros(t, batch_size, input_size))
    word = Variable(torch.ones(t, batch_size, input_size))

    #input_stack = torch.cat((landmark,word),1)    

    last_output= model(landmark)
    #print(last_output)

