import torch
from scipy.linalg import hankel
from tqdm.auto import tqdm


def mono_LMS(x, d, N, lr, normalize=True):
    n_blocks = x.shape[-1] - N

    u = torch.zeros(N)
    w = torch.zeros(N)

    e = torch.zeros(x.shape[-1])
    y = torch.zeros(x.shape[-1])
    for n in tqdm(range(n_blocks)):
        u[1:] = u.clone()[:-1]
        u[0] = x[n]
        y[n] = u @ w
        e_t = d[n] - y[n]
        if normalize:
            norm = (u @ u) + 1e-3
            w = w + lr * u * e_t / norm
        else:
            w = w + lr * u * e_t
        e[n] = e_t
    return y


def mono_NBLMS(x, d, N, L, lr, normalize=True, beta=0.9):
    n_blocks = x.shape[-1] // L
    norm = torch.full([L], 1e-3)

    u = torch.zeros(L + N - 1)
    w = torch.zeros(N)

    e = torch.zeros(n_blocks * L)
    y = torch.zeros(n_blocks * L)

    for k in tqdm(range(n_blocks)):
        u[:-L] = u.clone()[L:]
        u[-L:] = x[k * L:(k + 1) * L]
        d_t = d[k * L:(k + 1) * L]
        u_t = torch.tensor(hankel(u[:L], u[-N:]))
        y_t = u_t @ w
        y[k * L:(k + 1) * L] = y_t
        e_t = d_t - y_t
        if normalize:
            norm = beta * norm + (1 - beta) * (torch.sum(u_t ** 2, dim=1))
        w = w + lr * (u_t.T / (norm + 1e-3) @ e_t) / L
        e[k * L:(k + 1) * L] = e_t
    return y


def mono_Kalman(x, d, N, delta=1e-4):
    n_blocks = x.shape[-1] - N

    u = torch.zeros(N)
    w = torch.zeros(N)
    Q = torch.eye(N) * delta
    P = torch.eye(N) * delta
    I = torch.eye(N)

    e = torch.zeros(n_blocks)
    y = torch.zeros(n_blocks)
    for k in tqdm(range(n_blocks)):
        u[1:] = u.clone()[:-1]
        u[0] = x[k]
        y[k] = u @ w
        e_t = d[k] - y[k]
        R = e_t ** 2 + 1e-10
        P_t = P + Q
        r = P_t @ u
        K = r / ((u @ r) + R + 1e-10)
        w = w + (K * e_t)
        P = (I - torch.outer(K, u)) @ P_t
        e[k] = e_t
    return e, y