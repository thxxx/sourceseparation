import k_diffusion as K
import torch
from .sample import sample_dpmpp_3m_sde

def generation(
    model,
    ae,
    text_conditioner,
    text,
    steps,
    cfg_scale,
    duration=3.0,
    sample_rate=44100,
    batch_size=1,
    device='cuda',
    disable=False,
    train_duration=10.0,
    audio_context=None
):
    latent_channels = 64
    noise = torch.randn([batch_size, latent_channels, int(sample_rate*train_duration) // ae.downsampling_ratio], device=device, dtype=torch.float32)
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    
    input_ids, attention_mask = text_conditioner([text], device=device)
    seconds_start = torch.tensor([[0]], dtype=torch.float32)
    seconds_total = torch.tensor([[duration]], dtype=torch.float32)
    
    input_ids = input_ids.to(torch.float32)
    attention_mask = attention_mask.to(torch.float32)
    
    cross_attention_inputs, cross_attention_masks, global_cond = model.get_context(input_ids, attention_mask, seconds_start, seconds_total)

    model_dtype = next(model.parameters()).dtype
    noise = noise.type(model_dtype)
    
    sigmas = K.sampling.get_sigmas_polyexponential(100, 0.3, 500, rho=1.0, device=device)
    
    denoiser = K.external.VDenoiser(model)
    
    x = noise * sigmas[0]
    # print("noise : ", x)
    
    torch.set_printoptions(precision=10)  # 소수점 10자리까지 출력

    # with torch.cuda.amp.autocast():
    with torch.inference_mode():
        out = sample_dpmpp_3m_sde(
            denoiser, 
            x, 
            sigmas, 
            cross_attention_inputs,
            cross_attention_masks,
            seconds_start,
            seconds_total,
            disable=disable, 
            cfg_scale=cfg_scale, 
            callback=None,
            audio_context=audio_context
        )
        out = out.to(next(ae.parameters()).dtype)
        audio = ae.decode(out)
    
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak
    audio = audio.clamp(-1, 1)
    
    return audio
