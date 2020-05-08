import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_matrix_loss(input, target):
    dis_i = F.pdist(input, p=2)
    dis_t = F.pdist(target, p=2)

    loss = F.mse_loss(dis_i, dis_t)

    return loss

if __name__ == '__main__':
    a = torch.randn(5,5)
    b = torch.randn_like(a)

    print(cos_matrix_loss(a,b))