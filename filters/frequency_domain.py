import torch
from torch.fft import rfft as fft
from torch.fft import irfft as ifft

from tqdm.auto import tqdm


def mono_FDAF(x, d, M, lr=0.05, ff=0.9):
    norm = torch.full([M + 1], 1e-8)
    n_blocks = x.shape[-1] // M
    hann_window = torch.hann_window(M)
    w_fft = torch.zeros(M + 1)
    x_prev = torch.zeros(M)

    y = torch.zeros(n_blocks * M)

    for k in tqdm(range(n_blocks)):
        x_cur = torch.cat((x_prev, x[k * M:(k + 1) * M]))
        d_t = d[k * M:(k + 1) * M]

        x_prev = x[k * M:(k + 1) * M]

        x_fft = fft(x_cur)
        y_t = ifft(w_fft * x_fft)[M:]

        e_t = d_t - y_t

        y[k * M:(k + 1) * M] = y_t
        e_fft = fft(torch.cat((torch.zeros(M), e_t * hann_window)))

        norm = ff * norm + (1 - ff) * torch.abs(x_fft) ** 2
        f = lr * e_fft / (norm + 1e-3)
        w_fft = w_fft + f * x_fft.conj()
        w = ifft(w_fft)
        w[M:] = 0
        w_fft = fft(w)
    return y


def mono_FDKalman(x, d, M, ff=0.95, delta=1e-2, gamma=1e-6):
    n_blocks = x.shape[-1] // M
    hann_window = torch.hann_window(M)
    w_fft = torch.zeros(M + 1)
    x_prev = torch.zeros(M)

    Q = delta
    R = torch.full([M + 1], delta)
    P = torch.full([M + 1], gamma)

    y = torch.zeros(n_blocks * M)

    for k in tqdm(range(n_blocks)):
        x_cur = torch.cat((x_prev, x[k * M:(k + 1) * M]))

        d_t = d[k * M:(k + 1) * M]
        x_prev = x[k * M:(k + 1) * M]

        x_fft = fft(x_cur)

        y_t = ifft(w_fft * x_fft)[M:]

        e_t = d_t - y_t

        e_fft = fft(torch.cat((torch.zeros(M), e_t * hann_window)))

        R = ff * R + (1.0 - ff) * (torch.abs(e_fft) ** 2)
        P_t = P + Q * torch.abs(w_fft)
        K = P_t * x_fft.conj() / ((x_fft * P_t * x_fft.conj()) + R)

        P = (1.0 - K * x_fft) * P_t

        w_fft = w_fft + (K * e_fft)
        w = ifft(w_fft)
        w[M:] = 0
        w_fft = fft(w)

        y[k * M:(k + 1) * M] = y_t

    return y