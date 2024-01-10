import torch
import torch.nn as nn

"""
barch normalization 操作
"""
def batch_norm(is_training, X, gamma, beta, moving_mean,
                 moving_var, eps, momentum):
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        feature_shape = len(X.shape)
        assert feature_shape in (2, 4)

        if feature_shape == 2:
            # 全连接层后面的BN
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层后面的BN
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X-mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X-mean) / torch.sqrt(var + eps)
        # print(mean.shape)
        # print(var.shape)
        # print(X.shape)

        # 一阶指数平滑
        moving_mean = momentum * moving_mean + (1. - momentum) * mean
        moving_var = momentum * moving_var + (1. -momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class _BatchNorm(nn.Module):
    """
    BN层
    """
    def __init__(self, num_features, num_dims, momentum):
        super(_BatchNorm, self).__init__()
        assert num_dims in (2, 4)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
        self.momentum = momentum

    def forward(self, X):
        Y, self.moving_mean, self.moving_var = batch_norm(
            self.training, X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=self.momentum
        )
        return Y


class BatchNorm1d(_BatchNorm):
    def __init__(self, num_features, momentum=0.9):
        super().__init__(num_features, 2, momentum)


class BatchNorm2d(_BatchNorm):
    def __init__(self, num_features, momentum=0.9):
        super().__init__(num_features, 4, momentum)



# testing
X = torch.randn(32,128,32,32)
mean= X.mean(dim=0)
var = ((X - mean)**2).mean(dim=0)
net = BatchNorm2d(128)
net.train(True)
Y = net(X)
print(Y.shape)