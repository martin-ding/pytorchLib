import torch
import torch.nn.functional as F
# x = torch.randn(4, 784)
# w = torch.randn(10,784)
#
# logits = x@w.t()
# pred = F.softmax(logits, dim=1)
# print(pred)
# pred_log = torch.log(pred)
# print(torch.log(torch.tensor([1/2])))
# print(pred_log)
# print(F.nll_loss(pred_log, torch.tensor([3,1,3,1])))
#
# print(F.cross_entropy(logits, torch.tensor([3,1,3,1])))

print(torch.tensor([ 1.7659, -0.9089,  0.6360, -1.0340,  1.3676, -0.1486]).mean())