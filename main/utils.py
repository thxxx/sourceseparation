import numpy as np
import scipy.special
import scipy.stats
import torch
from einops import rearrange, repeat, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

AudioTensor = Float[Tensor, "batch audio audio_channel"]
AudioMaskTensor = Bool[Tensor, "batch audio"]
AudioChannelTensor = Float[Tensor, "batch 1 audio"]
AudioSegmentTensor = Float[Tensor, "batch 1 segment"]
EncTensor = Float[Tensor, "batch codec channel"]
EncMaskTensor = Bool[Tensor, "batch codec"]
EncodecCodeTensor = Int[Tensor, "batch code codec"]
LengthTensor = Int[Tensor, "batch"]
LossTensor = Float[Tensor, ""]
RotaryTensor = Float[Tensor, "codec dim_head"]
TimeTensor = Float[Tensor, "batch"]
Batch = tuple[AudioTensor, AudioMaskTensor]

def prob_mask_like(
    shape: torch.Size | tuple[int, ...] | list[int], prob: float, device: torch.device
):
    match prob:
        case 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        case 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        case _:
            return torch.rand(shape, device=device) < prob # prob 확률로 True = Mask

def mask_from_fracs(lengths: LengthTensor, fmin: float, fmax: float, max_len: int):
    batch = lengths.shape[0]
    fracs = torch.zeros(batch, device=lengths.device).float().uniform_(fmin, fmax)
    frac_lengths = (lengths.float() * fracs).clamp(min=1).round().long()
    max_starts = lengths - frac_lengths
    start_fracs = torch.zeros(batch, device=lengths.device).float().uniform_(0, 1)
    starts = (max_starts * start_fracs).round().long()
    ends = starts + frac_lengths

    seq = torch.arange(max_len, device=lengths.device)
    seq = repeat(seq, "l -> b l", b=lengths.shape[0])

    mask = (seq >= rearrange(starts, "b -> b ()")) & (
        seq < rearrange(ends, "b -> b ()")
    )
    return mask

def mask_from_lengths(lengths: LengthTensor, max_len: int) -> EncMaskTensor:
    seq = torch.arange(max_len, device=lengths.device)
    mask = seq < rearrange(lengths, "b -> b ()")
    return mask

def constant_slice_mask(
    lengths: LengthTensor, shape: int, max_len: int
) -> EncMaskTensor:
    max_start = lengths - lengths.clamp(max=max_len)
    starts = torch.rand(lengths.shape[0], device=lengths.device)
    starts = (starts * max_start).round().long()
    ends = starts + max_len
    seq = torch.arange(shape, device=lengths.device)
    seq = repeat(seq, "l -> b l", b=lengths.shape[0])
    mask = (seq >= rearrange(starts, "b -> b ()")) & (
        seq < rearrange(ends, "b -> b ()")
    )
    return mask


def num_windows(perc: float, length: int, min_span: int):
    max_window = int(np.floor(min(perc * length / min_span, (1 - perc) * length + 1)))
    if max_window == 1:
        return 1
    count = np.arange(1, max_window)
    op1 = scipy.special.loggamma(perc * length - count * min_span + count)
    op2 = scipy.special.loggamma(perc * length - count * min_span + 1)
    op3 = scipy.special.loggamma(count)
    op4 = scipy.special.loggamma((1 - perc) * length)
    op5 = scipy.special.loggamma((1 - perc) * length - count)
    op6 = scipy.special.loggamma(count + 1)
    log_prob = op1 - op2 - op3 + op4 - op5 - op6
    log_prob = scipy.special.softmax(log_prob, axis=0)
    rng = np.random.default_rng()
    return rng.choice(count.shape[0], p=log_prob) + 1


def min_span_mask(length: int, fmin: float, fmax: float, min_span: int):
    rng = np.random.default_rng()
    if min_span * 2 > length:
        min_span = round(length / 2)
    frac = rng.uniform(fmin, fmax)
    frac_length = round(frac * length)
    windows = num_windows(frac, length, min_span)
    window_sum = frac_length - windows * min_span + windows - 1
    blank_sum = length - frac_length + 1
    window_lengths = rng.choice(window_sum, windows - 1, replace=False)
    blank_lengths = rng.choice(blank_sum, windows, replace=False)
    window_lengths = np.concatenate([window_lengths, np.array([-1, window_sum])])
    blank_lengths = np.concatenate([blank_lengths, np.array([-1, blank_sum])])
    window_lengths.sort()
    blank_lengths.sort()
    window_lengths = np.diff(window_lengths) - 1
    blank_lengths = np.diff(blank_lengths) - 1
    window_lengths += min_span
    blank_lengths[1:-1] += 1
    total_lengths = np.zeros((2 * windows + 1), dtype=np.int64)
    total_lengths[1::2] = window_lengths
    total_lengths[::2] = blank_lengths
    binary_segments = np.arange(2 * windows + 1) % 2
    return binary_segments.repeat(total_lengths).astype(bool)


def get_span_mask(audio_feature, max_audio_len):
    mask_fracs = (0.1, 0.3)
    min_span = 4 # Encodec latent로 할 때 10이었으니 21.5Hz인 stable audio vae의 경우에는 4여도 되지 않을까?

    audio_lens = [audio_feature.shape[-1]]
    # audio_lens = audio_mask.sum(dim=1).detach().cpu().numpy()
    span_mask = pad_sequence(
        [
            torch.from_numpy(
                min_span_mask(
                    int(audio_len),
                    fmin=mask_fracs[0],
                    fmax=mask_fracs[1],
                    min_span=min_span,
                )
            ).to(audio_feature.device)
            for audio_len in audio_lens
        ],
        batch_first=True,
    )
    return F.pad(span_mask, (0, max_audio_len - span_mask.shape[1]))