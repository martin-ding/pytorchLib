import torch

# logits = torch.randn(3, 4, requires_grad=True)
# logits = torch.tensor([[ 1.4190,  0.2744,  1.7356, -0.1860],
#         [-0.7160, -0.6057,  0.2651, -1.1337],
#         [ 1.0048,  0.8326, -0.0885,  0.4471]], requires_grad= True)
#
# logits2 = torch.tensor([0.3980, 0.8603, 0.1073]);
#
# b = torch.log(torch.softmax(logits2, dim=0))
# print(b)
# labels = torch.LongTensor([1, 0, 2])
# print('logits={}, labels={}'.format(logits, labels))
#
#
# # 直接计算交叉熵（cross entropy loss）
# def calc_ce_loss1(logits, labels):
#     ce_loss = torch.nn.CrossEntropyLoss()
#     loss = ce_loss(logits, labels)
#     return loss
#
#
# # 分解计算交叉熵（cross entropy loss = log softmax + nll loss）
# def calc_ce_loss2(logits, labels):
#     log_softmax = torch.nn.LogSoftmax(dim=1)
#     nll_loss = torch.nn.NLLLoss() # Negative Log Likelihood Loss，也就是最大似然函数。
#     logits_ls = log_softmax(logits)
#     print(logits_ls)
#     loss = nll_loss(logits_ls, labels)
#     return loss
#
#
# loss1 = calc_ce_loss1(logits, labels)
# print('loss1={}'.format(loss1))
# loss2 = calc_ce_loss2(logits, labels)
# print('loss2={}'.format(loss2))
#
# # 增加 temperature
# temperature = 0.05
# logits_t = logits / temperature
# loss1 = calc_ce_loss1(logits_t, labels)
# print('t={}, loss1={}'.format(temperature, loss1))
# loss2 = calc_ce_loss2(logits_t, labels)
# print('t={}, loss2={}'.format(temperature, loss2))
#
# temperature = 2
# logits_t = logits / temperature
# loss1 = calc_ce_loss1(logits_t, labels)
# print('t={}, loss1={}'.format(temperature, loss1))
# loss2 = calc_ce_loss2(logits_t, labels)
# print('t={}, loss2={}'.format(temperature, loss2))


print("_____using BCELoss______")

#这里即可以看成3个样本，也可以看成一个样本对应的三个label
logits = torch.tensor([0.3,0.5,0.4]) # RuntimeError: all elements of input should be between 0 and 1
lossfun = torch.nn.BCELoss(reduction="none")
labels = torch.tensor([1,0,1]).type(torch.float32)
print(lossfun(logits, labels))

#一般使用，因为要先将logits所有的值变成0-1之间的数
# torch.nn.BCEWithLogitsLoss()  = torch.sigmoid() + torch.nn.BCELoss
