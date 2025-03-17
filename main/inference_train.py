import os
import math
import torch
from dit_main import DiffusionTransformer
from condition.t5condition import T5Conditioner
from inference.inference import generation
from train.dataset import SampleDataset, Config
from config.model.config import config
from vae.get_function import create_autoencoder_from_config
from train.utils import draw_plot, cleanup_memory, make_html
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda'

model_config = config['model']

if __name__ == "__main__":
    output_dir   = 'generation_inference'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    print("Start training")
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
        use_skip         = False,
        context_dim      = 64
    )
    model = model.to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n", num_trainable_params, "\n")

    pre_model_dir = '/home/khj6051'

    ae = create_autoencoder_from_config(config['auto_encoder_config']).to(device)
    ae_state_dict = torch.load(f'{pre_model_dir}/vae_weight.pth')
    ae_clean_state_dict = {layer_name.replace('model.', ''): weights for layer_name, weights in ae_state_dict.items()}
    ae.load_state_dict(ae_clean_state_dict)
    
    del ae_clean_state_dict
    del ae_state_dict
    cleanup_memory()
    
    text_conditioner = T5Conditioner(output_dim=768).to(device)
    model.load_state_dict(torch.load(f'./weights_0315/model_5.pth'), strict=False)
    
    ae.eval()
    text_conditioner.eval()
    
    batch_size   = 4
    lr           = 0.0001
    weight_decay = 0.001
    betas        = (0.9, 0.999)
    diffusion_objective = 'v'
    sample_rate  = 44100

    train_duration  = 10.0
    num_workers = 12
    num_epochs  = 100
    sample_cfg = {
        "sample_steps": 100,
        "sample_cfg": 6.0,
        "sample_duration": 8.0,
        "sample_rate": 44100,
        "diffusion_objective": "v"
    }

    config = Config(
        train_dataset='/home/khj6051/train_mix.csv',
        valid_dataset='/home/khj6051/valid_mix.csv', 
        duration=train_duration
    )

    valid_dataset = SampleDataset(config, mode="valid", sample_rate=sample_rate, force_channels='stereo')
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=True,  # Shuffle validation data without sampler
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    def get_alphas_sigmas(t):
        """Returns the scaling factors for the clean image (alpha) and for the
        noise (sigma), given a timestep."""
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
    
    trainer = {
        'train_losses': [],
        'train_losses_nocaption': [],
        'train_losses_noscript': [],
        'valid_losses': [],
        'sfx_valid_loss': [],
        'valid_images': [],
        'clap_score': [],
        'lrs': [],
    }
    scaler = torch.amp.GradScaler()

    torch.backends.cudnn.benchmark = True
    cleanup_memory()

    log_start_epoch = 0
    save_epoch = 1

    print("start training")
    model.eval()

    print(valid_dataset[1112])
    
    evals = []
    for vidx in [1112, 2223, 3334, 5556]:
        audio = valid_dataset[vidx][0]
        prompt = valid_dataset[vidx][1]['caption']
        duration = valid_dataset[vidx][1]['seconds_total']
        mixed_audio = valid_dataset[vidx][1]['mixed_audio']
        mixed_z_0 = ae.encode_audio(mixed_audio.to(device)) # B, C, SL
        
        output = generation(
            model,
            ae,
            text_conditioner,
            text        = prompt,
            steps       = sample_cfg['sample_steps'],
            cfg_scale   = sample_cfg['sample_cfg'],
            duration    = duration,
            sample_rate = sample_cfg['sample_rate'],
            batch_size  = 1,
            device      = device,
            disable     = False,
            audio_context = mixed_z_0
        )
        # torchaudio.save(f'{output_dir}/generated_{epoch}_{vidx}_{prompt}.wav', output.cpu()[0], sample_rate=44100)
        evals.append({
            'caption': 'mixed : ' + prompt,
            'array': mixed_audio
        })
        evals.append({
            'caption': 'target : ' + prompt,
            'array': audio
        })
        evals.append({
            'caption': 'generated : ' + prompt,
            'array': output.cpu()[0]
        })
    
    make_html(0, output_dir, evals)

