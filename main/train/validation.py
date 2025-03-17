import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from .utils import make_html, cleanup_memory
import torchaudio
import numpy as np
from jiwer import wer
from einops import rearrange, repeat, reduce

def masked_loss(
    x, target, mask, loss_method
):
    match loss_method:
        case "mse":
            loss = F.mse_loss(x, target, reduction="none")
        case "mae":
            loss = F.l1_loss(x, target, reduction="none")
        case _:
            raise ValueError(f"loss method {loss_method} not supported")

    loss = reduce(loss, "b c l -> b l", "mean")
    return loss[mask].mean()

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def validate_or_test(model, valid_loader, ae, text_conditioner, get_alphas_sigmas, cfg, generation, epoch, output_dir, device="cuda"):
    model.eval()
    valid_loss = 0
    nan_num = 0
    for idx, (audio, info) in enumerate(valid_loader):
        with torch.no_grad():
            audio = audio.to(device)
            ma = info['mixed_audio'].to(device)

            z_0 = ae.encode_audio(audio)
            mixed_z_0 = ae.encode_audio(ma)
            
            text = info['caption']
            input_ids, attention_mask = text_conditioner(text, device=device)
            
            seconds_start = info['seconds_start'].to(device)
            seconds_total = info['seconds_total'].to(device)
            
            t = torch.sigmoid(torch.randn(z_0.shape[0]))
            alphas, sigmas = get_alphas_sigmas(t)
            alphas = alphas[:, None, None].to(device)
            sigmas = sigmas[:, None, None].to(device)
            t = t.to(device)
            
            noise = torch.randn_like(z_0, device=device)
            noised_inputs = z_0 * alphas + noise * sigmas
            
            if cfg['diffusion_objective'] == "v":
                targets = noise * alphas - z_0 * sigmas
            elif cfg['diffusion_objective'] == 'rectified_flow':
                targets = noise - z_0
            
            t=t.unsqueeze(dim=1)
            seconds_start = seconds_start.unsqueeze(dim=1)
            seconds_total = seconds_total.unsqueeze(dim=1)
            
            # with autocast():
            output = model(
                x=noised_inputs, 
                t=t,
                mask=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                seconds_start=seconds_start,
                seconds_total=seconds_total,
                cfg_dropout_prob=0.01,
                cfg_scale=None,
                audio_context=mixed_z_0
            )
            loss = F.mse_loss(output, targets)
            if torch.isnan(loss):
                nan_num += 1
                continue
            valid_loss += loss

    del loss
    del output
    cleanup_memory()
    
    return valid_loss.cpu().detach().item()/len(valid_loader)
