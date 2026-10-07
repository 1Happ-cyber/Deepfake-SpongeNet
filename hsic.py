import torch
import numpy as np
from torch.autograd import Variable, grad


def sigma_estimation_torch(X, Y):
    """ sigma from median distance """
    device = X.device
    D = distmat(torch.cat([X, Y]))

    triu_indices = torch.triu_indices(D.shape[0], D.shape[1], offset=1, device=device)
    Tri = D[triu_indices[0], triu_indices[1]]

    med = torch.median(Tri)
    if med <= 0:
        med = torch.mean(Tri)
    if med < 1e-2:
        med = torch.tensor(1e-2, device=device)
    return med.item()


def distmat(X):
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def kernelmat(X, sigma, k_type="gaussian"):
    m = int(X.size()[0])
    device = X.device

    H = torch.eye(m, device=device) - (1. / m) * torch.ones(m, m, device=device)

    if k_type == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size(1)
            Kx = torch.exp(-Dxx / variance)
        else:
            sx = sigma_estimation_torch(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx))

    elif k_type == "linear":
        Kx = torch.mm(X, X.t())

    Kxc = torch.mm(Kx, H)
    return Kxc


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2. * sigma * sigma))
    return torch.mean(X)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_normalized(x, y, sigma=None):
    m = int(x.size()[0])
    if x.device != y.device:
        y = y.to(x.device)

    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)

    Pxy = torch.trace(torch.mm(Kxc, Kyc)) / ((m - 1) ** 2)
    Pxx = torch.trace(torch.mm(Kxc, Kxc)) / ((m - 1) ** 2)
    Pyy = torch.trace(torch.mm(Kyc, Kyc)) / ((m - 1) ** 2)

    thehsic = Pxy / (torch.sqrt(Pxx * Pyy) + 1e-8)

    return thehsic


if __name__ == "__main__":
    n_samples = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=== 测试1: 低维独立数据 ===")
    X_indep_low = torch.from_numpy(np.random.randn(n_samples, 2)).float().to(device)
    Y_indep_low = torch.from_numpy(np.random.randn(n_samples, 2)).float().to(device)
    print(f"归一化HSIC: {hsic_normalized(X_indep_low, Y_indep_low):.6f}")

    print("=== 测试2: 低维相关数据 ===")
    X_dep_low = torch.from_numpy(np.random.randn(n_samples, 2)).float().to(device)
    Y_dep_low = X_dep_low + 0.1 * torch.randn(n_samples, 2).float().to(device)
    print(f"归一化HSIC: {hsic_normalized(X_dep_low, Y_dep_low):.6f}")

    print("=== 测试3: 高维独立数据 ===")
    X_high = torch.from_numpy(np.random.randn(n_samples, 96)).float().to(device)
    Y_high = torch.from_numpy(np.random.randn(n_samples, 96)).float().to(device)
    print(f"归一化HSIC: {hsic_normalized(X_high, Y_high):.6f}")

    print("=== 测试4: 高维相关数据 ===")
    X_high = torch.from_numpy(np.random.randn(n_samples, 96)).float().to(device)
    Y_high = X_high + 0.001 * torch.randn(n_samples, 96).float().to(device)
    print(f"归一化HSIC: {hsic_normalized(X_high, Y_high):.6f}")


