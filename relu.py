import torch
import torch.nn.functional as F

# a = torch.linspace(-100, 100, 10)
# print(a)
# print(torch.sigmoid(a))
# print(torch.relu(a))
# print(torch.tanh(a))
# print(F.sigmoid(a))


# a = torch.full([2],1.)
# b = torch.full([2],2.)
# print(a-b, torch.norm(b-a,1))
#
# x = torch.tensor([1,2,3,4], dtype=torch.float32)
# print(x)
# w = torch.full([1], 3., requires_grad= True)
# # print(w.requires_grad_())
# print(w)
# mse = F.mse_loss(w * x, torch.tensor([1,2,3,4], dtype=torch.float32))
#
# print(mse)
# # print(torch.autograd.grad(mse, [w]))
# mse.backward()
# print(w.grad)


# a = torch.rand(3, requires_grad=True)
# p = F.softmax(a, dim= 0)
# print(p)
# p[0].backward(retain_graph=True)
# print(a.grad)
# p[0].backward()
# print(a.grad)


# x = torch.randn([1,10])
# w = torch.randn([2,10], requires_grad=True)
# loss = F.mse_loss(F.sigmoid(x@w.t()), torch.ones([1,2]))
# loss.backward()
# print(w.grad)


# x = torch.tensor(1.)
# w1 = torch.tensor(2., requires_grad=True)
# b1 = torch.tensor(1.,requires_grad=True)
# w2 = torch.tensor(2., requires_grad=True)
# b2 = torch.tensor(1., requires_grad=True)
# y1 = x * w1 + b1
# y2 = y1 * w2 + b2
#
# dy2_y1 = torch.autograd.grad(y2, [y1], retain_graph=True)
# dy2_w = torch.autograd.grad(y2, [w1], retain_graph=True)
# dy1_w = torch.autograd.grad(y1, [w1], retain_graph=True)
# dy2_b1 = torch.autograd.grad(y2, [b1], retain_graph=True)
#
# print(dy2_y1, dy2_w, dy1_w, dy2_b1)

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 1)
y = np.arange(-6, 6, 1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X, Y)
Z = himmelblau([X, Y])
fig = plt.figure('himmelblau')
# ax = fig.gca(projection='3d')
ax = fig.add_axes(Axes3D(fig))
ax.plot_surface(X, Y, Z)
ax.view_init(20, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#
# # [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([-4., 0.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):

    pred = himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print ('step {}: x = {}, f(x) = {}'
               .format(step, x.tolist(), pred.item()))