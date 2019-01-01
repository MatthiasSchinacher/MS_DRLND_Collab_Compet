# * ---------------- *
#
#   ** Deep Reinforcement Learning Nano Degree **
#   project: Collaboration and Competition
#   author:  Matthias Schinacher
#
# small helper script, to output a model
# * ---------------- *

# * ---------------- *
#    importing the packages we need
# * ---------------- *
import os.path
import sys
import torch

class MSA(torch.nn.Module):
    def __init__(self, action_size, state_size, size1=111, size2=87, flag_batch_norm=False):
        super(MSA, self).__init__()
        self.ll1 = torch.nn.Linear(state_size, size1)
        self.r1  = torch.nn.ReLU()
        self.ll2 = torch.nn.Linear(size1, size2)
        self.r2  = torch.nn.ReLU()
        self.ll3 = torch.nn.Linear(size2, action_size)
        self.th  = torch.nn.Tanh()
        
        self.flag_batch_norm = flag_batch_norm
        if flag_batch_norm:
            self.batch = torch.nn.BatchNorm1d(state_size)

        torch.nn.init.uniform_(self.ll1.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll1.bias,0.1)
        torch.nn.init.uniform_(self.ll2.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll2.bias,0.1)
        torch.nn.init.uniform_(self.ll3.weight,-0.001,0.001)
        torch.nn.init.constant_(self.ll3.bias,0.1)

    def forward(self, state):
        if self.flag_batch_norm:
            return self.th(self.ll3(self.r2(self.ll2(self.r1(self.ll1(self.batch(state)))))))
        else:
            return self.th(self.ll3(self.r2(self.ll2(self.r1(self.ll1(state))))))

class MSC(torch.nn.Module):
    def __init__(self, action_size, state_size, size1=111, size2=87, flag_batch_norm=False):
        super(MSC, self).__init__()
        self.ll1 = torch.nn.Linear(state_size, size1)
        self.r1  = torch.nn.ReLU()
        self.ll2 = torch.nn.Linear(size1+action_size, size2)
        self.r2  = torch.nn.ReLU()
        self.ll3 = torch.nn.Linear(size2, action_size)
        
        self.flag_batch_norm = flag_batch_norm
        if flag_batch_norm:
            self.batch = torch.nn.BatchNorm1d(state_size)

        torch.nn.init.uniform_(self.ll1.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll1.bias,0.1)
        torch.nn.init.uniform_(self.ll2.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll2.bias,0.1)
        torch.nn.init.uniform_(self.ll3.weight,-0.001,0.001)
        torch.nn.init.constant_(self.ll3.bias,0.1)

    def forward(self, state, action):
        x = state
        if self.flag_batch_norm:
            x = self.r1(self.ll1(self.batch(x)))
            return self.ll3(self.r2(self.ll2(torch.cat((x, action), dim=1))))
        else:
            x = self.r1(self.ll1(x))
            return self.ll3(self.r2(self.ll2(torch.cat((x, action), dim=1))))

# * ---------------- *
#   command line arguments:
#    we expect exactly 2, the actual script name and the model-file-name
# * ---------------- *
if len(sys.argv) != 2:
    print('usage:')
    print('   python {} model-file-name'.format(sys.argv[0]))
    quit()

if not os.path.isfile(sys.argv[1]):
    print('usage:')
    print('   python {} model-file-name'.format(sys.argv[0]))
    print('[error] "{}" file not found or not a file!'.format(sys.argv[1]))
    quit()

modelQ = torch.load(sys.argv[1])
print('model from file "{}:"\n'.format(sys.argv[1]))
#print(modelQ)
for x in modelQ.modules():
    #print(type(x))
    print(x)
    if isinstance(x, torch.nn.modules.linear.Linear):
        print('weight:',x.weight)
        print('bias:',x.bias)
