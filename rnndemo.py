import torch
import torch.nn as nn

rnn = nn.RNN(100, 10, num_layers= 2)
print(list(rnn._parameters.keys()))
for name, w in rnn.named_parameters() :
    print(name, '\n', w.shape)

x = torch.randn(5,3,100)
out, h = rnn(x)
print(out.shape, h.shape)
# print(out[4], h[1])