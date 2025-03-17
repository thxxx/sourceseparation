import torch
from dit_main import DiffusionTransformer
from config.model.config import config
from condition.t5condition import T5Conditioner

device = 'cuda'

model = DiffusionTransformer(
    d_model          = 1536,
    depth            = 24,
    num_heads        = 24,
    input_concat_dim = 0,
    global_cond_type = 'prepend',
    latent_channels  = 64, 
    config           = config, 
    device           = device,
    ada_cond_dim     = None,
    use_skip         = False
)
model = model.to(device)
model.eval()

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_trainable_params)

pre_model_dir = '/home/khj6051/mel_con_sample/pretrained2'

state_dict = torch.load(f'{pre_model_dir}/transformer_weight.pth')

clean_state_dict = {layer_name.replace('ff.ff.0', 'ff.linear_in'): weights for layer_name, weights in state_dict.items()}
clean_state_dict = {layer_name.replace('ff.ff.2', 'ff.linear_out'): weights for layer_name, weights in clean_state_dict.items()}

model.model.load_state_dict(clean_state_dict)

s1 = torch.load(f'{pre_model_dir}/preprocess_conv.pth')
s2 = torch.load(f'{pre_model_dir}/postprocess_conv.pth')
s3 = torch.load(f'{pre_model_dir}/to_cond_embed.pth')
s4 = torch.load(f'{pre_model_dir}/to_global_embed.pth')
s5 = torch.load(f'{pre_model_dir}/to_timestep_embed.pth')

del clean_state_dict
del state_dict
torch.cuda.empty_cache()

model.preprocess_conv.load_state_dict(s1)
model.postprocess_conv.load_state_dict(s2)
model.to_cond_embed.load_state_dict(s3)
model.to_global_embed.load_state_dict(s4)
model.to_timestep_embed.load_state_dict(s5)

ss = torch.load(f'{pre_model_dir}/sec_start.pth')
model.timing_start_conditioner.load_state_dict(ss)

st = torch.load(f'{pre_model_dir}/sec_total.pth')
model.timing_total_conditioner.load_state_dict(st)

model.fourier.load_state_dict(torch.load(f'{pre_model_dir}/timestep_features.pth'))

del ss
del st
del s1
del s2
del s3
del s4
del s5
torch.cuda.empty_cache()

torch.save(model.state_dict(), './stable_audio_origin_weight.pth')