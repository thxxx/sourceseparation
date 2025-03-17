import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dit_main import DiffusionTransformer
from condition.t5condition import T5Conditioner
from train.validation import validate_or_test
from inference.inference import generation
from train.dataset import SampleDataset, Config
from config.model.config import config
from vae.get_function import create_autoencoder_from_config
from train.utils import draw_plot, cleanup_memory, calculate_targets, prepare_batch_data, make_html
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import Dataset, DataLoader, RandomSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch._dynamo.config.cache_size_limit = 128
device = 'cuda'

model_config = config['model']

if __name__ == "__main__":
    output_dir   = 'weights_0315'
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
    torch.cuda.empty_cache()
    
    text_conditioner = T5Conditioner(output_dim=768).to(device)
    model.load_state_dict(torch.load(f'./weights_0313/model_2.pth'), strict=False)
    
    ae.eval()
    text_conditioner.eval()
    
    batch_size   = 32
    lr           = 0.00004
    weight_decay = 0.001
    betas        = (0.9, 0.999)
    diffusion_objective = 'v'
    sample_rate  = 44100

    train_duration  = 10.0
    num_workers = 10
    num_epochs  = 100
    sample_cfg = {
        "sample_steps": 100,
        "sample_cfg": 6.5,
        "sample_duration": 8.0,
        "sample_rate": 44100,
        "diffusion_objective": "v"
    }

    config = Config(
        train_dataset='/home/khj6051/mel_con_sample/train_mix.csv',
        valid_dataset='/home/khj6051/mel_con_sample/valid_mix.csv', 
        duration=train_duration
    )

    steps_per_epoch = 1999
    num_samples_per_epoch = steps_per_epoch * batch_size  # 총 샘플 수 = 스텝 수 * 배치 크기 # train 320,000 samples per epoch

    train_dataset = SampleDataset(config, mode="train", sample_rate=sample_rate, force_channels='stereo')
    valid_dataset = SampleDataset(config, mode="valid", sample_rate=sample_rate, force_channels='stereo')
    # Define Train Sampler
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples_per_epoch)  # Train Sampler with replacement

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,  # Shuffle validation data without sampler
        num_workers=num_workers,
        persistent_workers=False,
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

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

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
    for epoch in range(num_epochs):
        model.train()
        
        epoch_loss = 0
        tqdm_bar = tqdm(total=len(train_loader), desc="Diffusion Training")
        for idx, (audio, info) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # 데이터 전처리 부분을 함수로 분리
            # with torch.amp.autocast():
            z_0, mixed_z_0, input_ids, attention_mask, mean_pooled = prepare_batch_data(
                audio, info, ae, text_conditioner, device
            )
            
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
            
            loss.backward()
            optimizer.step()
            
            tqdm_bar.update()
            epoch_loss += loss

            if idx%500 == 499:
                scheduler.step()
                trainer['train_losses'].append(epoch_loss.cpu().detach().item()/idx)
                trainer['lrs'].append(optimizer.param_groups[0]['lr'])
                
                draw_plot('train_losses', trainer, output_dir=output_dir)
                draw_plot('lrs', trainer, output_dir=output_dir)
                # 텍스트 파일에 쓰기
                with open(f'./{output_dir}/middle_logs.txt', 'a') as file:
                    file.write(f"\nEpoch - {epoch} {epoch_loss.cpu().detach().item()/idx}\n")
                torch.cuda.empty_cache()
                torch.save(model.state_dict(), f'./{output_dir}/model_{epoch}.pth')
        
        if epoch>=log_start_epoch:
            scheduler.step()
            trainer['train_losses'].append(epoch_loss.cpu().detach().item()/idx)
            trainer['lrs'].append(optimizer.param_groups[0]['lr'])            
            draw_plot('train_losses', trainer, output_dir=output_dir)
            draw_plot('lrs', trainer, output_dir=output_dir)
        
        if epoch%save_epoch == save_epoch-1:
            torch.save(model.state_dict(), f'./{output_dir}/model_{epoch}.pth')

        del loss
        del noised_inputs
        del output
        torch.cuda.empty_cache()

        # 학습 끝
        model.eval()
        sfx_valid_loss = 0
        try:
            sfx_valid_loss = validate_or_test(
                model=model, 
                valid_loader=valid_loader, 
                ae=ae, 
                text_conditioner=text_conditioner, 
                get_alphas_sigmas=get_alphas_sigmas, 
                generation=generation,
                cfg=sample_cfg,
                epoch=epoch,
                output_dir=output_dir,
                device=device
            )
        except Exception as e:
            print("Exception : ", e)

        evals = []
        for vidx in [111, 2222, 3333, 5555, 8888]:
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
        make_html(epoch, output_dir, evals)
        
        if epoch>=log_start_epoch:
            trainer['valid_losses'].append(sfx_valid_loss)
            draw_plot('valid_losses', trainer, output_dir=output_dir)

        # 텍스트 파일에 쓰기
        with open(f'./{output_dir}/logs.txt', 'a') as file:
            file.write(f"\nEpoch - {epoch}\n")
            file.write(f"Train loss : {epoch_loss/len(train_loader)}\n")
            file.write(f"Valid loss : {sfx_valid_loss}\n")
        
        torch.cuda.empty_cache()

