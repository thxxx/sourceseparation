import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dit_main import DiffusionTransformer
from condition.t5condition import T5Conditioner
# from train.dataset import SampleDataset, Config
from train.dataset_group import SampleDataset, Config
from config.model.config import config
from vae.get_function import create_autoencoder_from_config
from train.utils import draw_plot, cleanup_memory, calculate_targets, prepare_batch_data, make_html
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import Dataset, DataLoader, RandomSampler
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch._dynamo.config.cache_size_limit = 128
device = 'cuda'

model_config = config['model']

if __name__ == "__main__":
    output_dir   = 'weights_0312_2'
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

    pre_model_dir = '/home/khj6051/mel_con_sample'

    ae = create_autoencoder_from_config(config['auto_encoder_config']).to(device)
    ae_state_dict = torch.load(f'{pre_model_dir}/vae_weight.pth')
    ae_clean_state_dict = {layer_name.replace('model.', ''): weights for layer_name, weights in ae_state_dict.items()}
    ae.load_state_dict(ae_clean_state_dict)
    
    del ae_clean_state_dict
    del ae_state_dict
    cleanup_memory()
    
    text_conditioner = T5Conditioner(output_dim=768).to(device)
    model.load_state_dict(torch.load(f'./weights_0311/model_2.pth'), strict=False)
    
    ae.eval()
    text_conditioner.eval()
    
    batch_size   = 4
    lr           = 0.0001
    weight_decay = 0.001
    betas        = (0.9, 0.999)
    diffusion_objective = 'v'
    sample_rate  = 44100

    train_duration  = 10.0
    num_workers = 8
    num_epochs  = 100
    sample_cfg = {
        "sample_steps": 100,
        "sample_cfg": 6.5,
        "sample_duration": 8.0,
        "sample_rate": 44100,
        "diffusion_objective": "v"
    }

    config = Config(
        train_dataset='/home/khj6051/train_mix.csv',
        valid_dataset='/home/khj6051/valid_mix.csv', 
        duration=train_duration
    )

    steps_per_epoch = 1999
    num_samples_per_epoch = steps_per_epoch * batch_size  # 총 샘플 수 = 스텝 수 * 배치 크기 # train 320,000 samples per epoch

    def custom_collate_fn(batch):
        """
        batch: [
            ([audio_1, audio_2, ..., audio_10], [info_1, info_2, ..., info_10]),
            ([audio_11, ..., audio_20], [info_11, ..., info_20]),
            ...
        ]
        """
        batch_audio = []
        batch_info = []

        for audio_list, info_list in batch:
            batch_audio.extend(audio_list)  # 리스트 풀기
            batch_info.extend(info_list)

        return torch.stack(batch_audio), batch_info  # audio는 텐서, info는 리스트 유지
    
    train_dataset = SampleDataset(config, mode="train", sample_rate=sample_rate, force_channels='stereo')
    # Define Train Sampler
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples_per_epoch)  # Train Sampler with replacement

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn  # 여기에 추가
    )
    for _ in range(3):
        dd = next(iter(train_loader))
        print(f"dd : \n\n{[d['audio_path'] for d in dd[1]]}\n\n")

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

    # scaler = torch.amp.GradScaler()

    torch.backends.cudnn.benchmark = True
    cleanup_memory()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=20, # 20 epoch마다 lr 감소
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=lr/10,
        warmup_steps=4,
        gamma=0.8 # 한 사이클 돌 때마다 max lr이 80%가 된다.
    )

    log_start_epoch = 0
    save_epoch = 1

    print("start training")
    nan_num = 0
    model.train()
    
    epoch_loss = 0
    tqdm_bar = tqdm(total=len(train_loader), desc="Diffusion Training")
    ae.encode_audio = torch.compile(ae.encode_audio, dynamic=True)

    end_time = time.time()

    loaders = 0
    aes = 0
    models = 0
    cals = 0

    ae1s = 0
    ae2s = 0
    texts = 0
    for idx, (audio, info) in enumerate(train_loader):
        before_ae_time = time.time()
        loader_latency = before_ae_time - end_time
        loaders += loader_latency
        optimizer.zero_grad(set_to_none=True)
        
        # 데이터 전처리 부분을 함수로 분리
        with torch.no_grad():
            audio = audio.to(device)
            start = time.time()
            z_0 = ae.encode_audio(audio) # B, C, SL
            torch.cuda.synchronize()
            t1 = time.time()
            ae1s += t1 - start

            torch.cuda.synchronize()
            start = time.time()
            ma = info['mixed_audio'].to(device)
            mixed_z_0 = ae.encode_audio(ma) # B, C, SL
            torch.cuda.synchronize()
            t2 = time.time()
            ae2s += t2 - start
            
            torch.cuda.synchronize()
            start = time.time()
            text = info['caption']
            input_ids, attention_mask = text_conditioner(text, device=device)
            valid_tokens = attention_mask.sum(dim=1, keepdim=True)
            mean_pooled = input_ids.sum(dim=1)/valid_tokens
            torch.cuda.synchronize()
            t3 = time.time()
            texts += t3 - start
        # z_0, mixed_z_0, input_ids, attention_mask, mean_pooled = prepare_batch_data(
        #     audio, info, ae, text_conditioner, device
        # )
        
        seconds_start = info['seconds_start'].unsqueeze(dim=1).to(device)
        seconds_total = info['seconds_total'].unsqueeze(dim=1).to(device)

        bs, ch, sl = z_0.shape
        t = torch.sigmoid(torch.randn(bs))
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None].to(device)
        sigmas = sigmas[:, None, None].to(device)
        t = t.unsqueeze(dim=1).to(device)
        
        noise = torch.randn_like(z_0, device=device)
        noised_inputs = z_0 * alphas + noise * sigmas
        
        targets = calculate_targets(noise, z_0, alphas, sigmas, diffusion_objective)
        
        before_model_time = time.time()
        ae_latency = before_model_time - before_ae_time
        aes += ae_latency
        output = model(
            x=noised_inputs, 
            t=t,
            mask=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            seconds_start=seconds_start,
            seconds_total=seconds_total,
            cfg_dropout_prob=0.2,
            cfg_scale=None,
            audio_context=mixed_z_0
        )
        loss = F.mse_loss(output, targets)

        if torch.isnan(loss):
            nan_num += 1
            continue
        
        before_backward_time = time.time()
        model_latency = before_backward_time - before_model_time
        models += model_latency

        loss.backward()
        optimizer.step()
        
        tqdm_bar.update()
        epoch_loss += loss

        end_time = time.time()
        cal_latency = end_time - before_backward_time
        cals += cal_latency

        if idx>50:
            break
    
    print(f"""
        loader latency : {loaders}\n    
        ae latency : {aes}\n    
        models latency : {models}\n    
        cal latency : {cals}\n      
        ae1s : {ae1s}\n    
        ae2s : {ae2s}\n    
        texts : {texts}\n    
    """)

