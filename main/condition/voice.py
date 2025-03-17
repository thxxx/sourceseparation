import librosa
import librosa.display
import matplotlib.pyplot as plt
from audiotools import AudioSignal
import torch
from librosa import filters
import torchaudio
from einops import rearrange
import numpy as np
import random

n_chroma = 12
radix2_exp = 13 # 13
winlen = 2 ** radix2_exp + 512
sample_rate = 44100
nfft = winlen
winhop = winlen // 4 # 4

fbanks = torch.from_numpy(filters.chroma(sr=sample_rate, n_fft=nfft, tuning=0, n_chroma=n_chroma))

spec = torchaudio.transforms.Spectrogram(
    n_fft=nfft, 
    win_length=winlen,
  hop_length=winhop, power=2, center=True,
  pad=0, normalized=True)

quant_levels = [
    # (0.00, 0.05, 0.0),
    # (0.05, 0.15, 0.1),
    # (0.15, 0.25, 0.2),
    # (0.25, 0.35, 0.3),
    # (0.35, 0.45, 0.4),
    # (0.45, 0.55, 0.5),
    # (0.55, 0.65, 0.6),
    # (0.65, 0.75, 0.7),
    # (0.75, 0.85, 0.8),
    # (0.85, 0.95, 0.9),
    (0.0, 0.99, 0.0),
    (0.99, 1.0, 1.0),
]

def quantize_tensor(tensor, quant_levels):
    quantized = torch.zeros_like(tensor)
    
    for min_val, max_val, target_value in quant_levels:
        mask = (tensor >= min_val) & (tensor <= max_val)
        quantized[mask] = target_value

    return quantized

def mask_random_columns(tensor, mask_ratios=(0.1, 0.4)):
    # assert tensor.shape == (64, 215), "Input tensor must have shape (64, 215)"
    mask_ratio = [0.1, 0.2, 0.3, 0.4][random.randint(0, 3)]

    num_columns = tensor.shape[1]  # 215
    num_masked = int(num_columns * mask_ratio)  # 마스킹할 column 개수 (10%면 약 21개)
    masked_indices = torch.randperm(num_columns)[:num_masked]

    tensor[:, masked_indices] = 0

    return tensor

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def make_voice_cond(audio):
    if len(audio.shape) == 3 and audio.shape[1] == 2: # if batch with stereo
        audio = (audio[:, 0, :] + audio[:, 1, :])/2
    elif len(audio.shape) == 2 and audio.shape[0] == 2:
        audio = (audio[0, :] + audio[1, :])/2
    
    ys = spec(audio)
    
    raw_chroma = torch.einsum('cf,...ft->...ct', fbanks, ys)
    norm_chroma = torch.nn.functional.normalize(raw_chroma, p=torch.inf, dim=-2, eps=1e-6)
    norm_chroma = quantize_tensor(norm_chroma, quant_levels)
    
    length = norm_chroma.shape[-1]

    rms = librosa.feature.rms(y=audio)
    l = rms.shape[-1]//length
    qrms = []
    for i in range(length):
        # batch로 처리할지 아닐지 여기서 갈린다. row[:, :, :] or row[:, :]
        qrms.append(np.sum(rms[:, max(i*l-3, 0) : (i+1)*l+3], axis=-1, keepdims=True))

    qrmss = torch.tensor(np.array(qrms)).squeeze().unsqueeze(dim=0)
    qrmss = qrmss.expand(4, -1)
    qrmss = min_max_normalize(qrmss)
    
    total = torch.cat((qrmss, norm_chroma), dim=-2)
    # total = rearrange(total, 'b d n -> b n d')
    # total = mask_random_columns(total)
    
    return total