import os
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import random
import librosa
import torchaudio
from tqdm import tqdm
from train.dataset import Config
from dit_clap import DiffusionTransformer
from condition.t5condition import T5Conditioner
from condition.clap_model import CLAPModule

from train.validation import validate_or_test
from inference.inference import generation
from train.dataset import SampleDataset
from config.model.config import config
from vae.get_function import create_autoencoder_from_config
from train.utils import draw_plot, cleanup_memory, save_concatenated_mel_spectrogram, calculate_targets, prepare_batch_data_with_cross, masked_loss
from utils import get_span_mask, prob_mask_like
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader, RandomSampler

from condition.voice import make_voice_cond

# ------------------------------
# Distributed Training Utilities
# ------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """
    초기 분산 학습 환경 설정 (NCCL backend와 env:// 방식 사용)
    """
    dist.init_process_group(backend='nccl', init_method='env://')

def cleanup_distributed():
    dist.destroy_process_group()

# ------------------------------
# 예제 오디오 리스트 (샘플 생성 시 사용)
# ------------------------------
audios = [
    {
        'audio_path': '/workspace/mel_con_sample/testsamples/bass_synth.wav',
        'prompt': 'prompts: bass, loop',
        'duration': 3.7
    },
    {
        'audio_path': '/workspace/mel_con_sample/testsamples/guitar_melody.wav',
        'prompt': 'prompts: guitar, melody, loop',
        'duration': 13.0
    },
    {
        'audio_path': '/workspace/mel_con_sample/testsamples/hiphop_vocal.wav',
        'prompt': 'prompts: hip-hop, vocal, loop, wet',
        'duration': 6.9
    },
    {
        'audio_path': '/workspace/mel_con_sample/testsamples/stars.wav',
        'prompt': 'prompts: live, vocal, one shot, wet',
        'duration': 3.35
    },
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch._dynamo.config.cache_size_limit = 128

if __name__ == "__main__":
    # 분산 학습 초기화 및 현재 프로세스(GPU) 할당
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    if dist.get_rank() == 0:
        print("Start training")
    
    # ------------------------------
    # 모델, VAE, 조건자 초기화
    # ------------------------------
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
        is_melody_prompt = True,
        is_audio_prompt  = True
    )
    model = model.to(device)
    # DDP 래핑 (모델 파라미터 동기화)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
    find_unused_parameters=True)
    # model = torch.compile(model)
    
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.get_rank() == 0:
        print("\nNumber of trainable params:", num_trainable_params, "\n")
    
    pre_model_dir = '/workspace/mel_con_sample'
    
    ae = create_autoencoder_from_config(config['auto_encoder_config']).to(device)
    ae_state_dict = torch.load(f'{pre_model_dir}/vae_weight.pth', map_location=device)
    ae_clean_state_dict = {layer_name.replace('model.', ''): weights for layer_name, weights in ae_state_dict.items()}
    ae.load_state_dict(ae_clean_state_dict)
    del ae_clean_state_dict, ae_state_dict
    torch.cuda.empty_cache()
    ae.eval()
    
    text_conditioner = T5Conditioner(output_dim=768).to(device)
    text_conditioner.eval()
    
    clap_model = CLAPModule().to(device)
    clap_model.eval()
    
    model.load_state_dict(torch.load(f'{pre_model_dir}/total_model_350.pth'), strict=False)
    cleanup_memory()
    
    # ------------------------------
    # 하이퍼파라미터 및 DataLoader 설정
    # ------------------------------
    batch_size   = 32
    lr           = 0.0001
    weight_decay = 0.001
    betas        = (0.9, 0.999)
    diffusion_objective = 'v'
    sample_rate  = 44100
    train_duration  = 10.0
    num_workers = 32
    num_epochs  = 100
    sample_cfg = {
        "sample_steps": 100,
        "sample_cfg": 6.5,
        "sample_duration": 8.0,
        "sample_rate": 44100,
        "diffusion_objective": "v"
    }
    
    output_dir   = 'weights_0205_clap_full'
    if dist.get_rank() == 0 and not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    config_obj = Config(
        train_dataset='./train_0205.csv', 
        valid_dataset='./valid_16.csv', 
        duration=train_duration
    )
    
    # 원래는 steps_per_epoch×batch_size 만큼의 샘플을 사용하도록 되어 있음.
    steps_per_epoch = 9999
    num_samples_per_epoch = steps_per_epoch * batch_size  # Global total samples per epoch
    
    train_dataset = SampleDataset(config_obj, mode="train", sample_rate=sample_rate, force_channels='stereo')
    valid_dataset = SampleDataset(config_obj, mode="valid", sample_rate=sample_rate, force_channels='stereo')
    
    # 각 GPU마다 처리할 샘플 수 (global 샘플 수를 GPU 수로 나눔)
    world_size = dist.get_world_size()
    num_samples_per_rank = num_samples_per_epoch // world_size
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples_per_rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    # 검증은 rank 0에서만 진행 (다른 GPU에서는 생략)
    if dist.get_rank() == 0:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
    else:
        valid_loader = None
    
    def get_alphas_sigmas(t):
        """주어진 t에 대해 clean 이미지에 대한 스케일(alpha)와 noise에 대한 스케일(sigma)을 반환"""
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
    
    scaler = GradScaler()
    torch.backends.cudnn.benchmark = True
    cleanup_memory()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=30,  # 약 30 epoch마다 lr 감소
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=lr/10,
        warmup_steps=5,
        gamma=0.8  # 사이클마다 max lr의 80%
    )
    
    log_start_epoch = 0
    save_epoch = 1
    
    if dist.get_rank() == 0:
        print("start training")
    rate = ae.sample_rate / ae.downsampling_ratio
    
    # ------------------------------
    # 학습 루프
    # ------------------------------
    for epoch in range(num_epochs):
        model.train()
        # text_conditioner.proj_out.train()
        epoch_loss = 0.0
        nan_num = 0
        
        # rank 0에서만 진행상황을 tqdm으로 표시
        tqdm_bar = tqdm(total=len(train_loader), desc=f"Diffusion Training Epoch {epoch}") if dist.get_rank() == 0 else None
        
        for idx, (audio, info) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # 데이터 전처리: VAE, 텍스트/오디오 조건자 적용
            z_0, input_ids, attention_mask, mean_pooled, audio_embed = prepare_batch_data_with_cross(
                audio, info, ae, text_conditioner, device, clap_model
            )
            
            # (옵션) 일정 확률로 조건을 제거할 수 있음
            if random.random() < 0.2:
                audio_embed = torch.zeros_like(audio_embed).to(device)
            
            audio_path = info['audio_path']
            seconds_start = info['seconds_start'].unsqueeze(dim=1).to(device)
            seconds_total = info['seconds_total'].unsqueeze(dim=1).to(device)
            melody = info['melody'].to(device)
            
            bs, ch, sl = z_0.shape
            t = torch.sigmoid(torch.randn(bs, device=device))
            alphas, sigmas = get_alphas_sigmas(t)
            alphas = alphas[:, None, None].to(device)
            sigmas = sigmas[:, None, None].to(device)
            t = t.unsqueeze(dim=1).to(device)
            
            noise = torch.randn_like(z_0, device=device)
            noised_inputs = z_0 * alphas + noise * sigmas
            
            targets = calculate_targets(noise, z_0, alphas, sigmas, diffusion_objective)
            
            if random.random() < 0.2:
                melody = torch.zeros_like(melody)
            
            output = model(
                x=noised_inputs, 
                t=t,
                mask=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                seconds_start=seconds_start,
                seconds_total=seconds_total,
                cfg_dropout_prob=0.1,
                cfg_scale=None,
                melody_prompt=melody,
                audio_embed=audio_embed
            )
            loss = F.mse_loss(output, targets)
            
            # print(f"d- {dist.get_rank()} loss : ", loss)
            # if torch.isnan(loss):
            #     print(f"⚠️ NaN detected at epoch {epoch}, step {idx}, rank {dist.get_rank()} - Skipping step")
            #     cleanup_memory()
            #     dist.barrier()  # DDP 동기화
            #     continue  # 🛑 해당 step을 건너뜀

            # 🛑 NaN 발생 여부를 모든 GPU에서 확인
            is_nan = torch.isnan(loss).float()  # NaN이면 1.0, 아니면 0.0
            dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)  # 모든 GPU에서 합산
        
            if is_nan.item() > 0:  # 하나라도 NaN이면 skip
                if dist.get_rank() == 0:  # rank 0에서만 출력
                    print(f"⚠️ NaN detected at epoch {epoch}, step {idx} - Skipping step")
                cleanup_memory()
                dist.barrier()  # 모든 GPU 동기화
                nan_num += 1
                continue  # 🛑 해당 step 건너뛰기
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if tqdm_bar is not None:
                tqdm_bar.update()
            epoch_loss += loss.item()
            
            if idx % 50 == 49 and dist.get_rank() == 0:
                print(f"Epoch {epoch} Iter {idx}: loss {loss.item()}, nan count: {nan_num}")
                with open(f'./{output_dir}/middle_logs.txt', 'a') as file:
                    file.write(f"\nEpoch {epoch} Iter {idx}: {loss.item()} | Avg Loss: {epoch_loss/idx}\n")
            
            if idx % 500 == 499 and dist.get_rank() == 0:
                scheduler.step()
                trainer['train_losses'].append(epoch_loss/idx)
                trainer['lrs'].append(optimizer.param_groups[0]['lr'])
                draw_plot('train_losses', trainer, output_dir=output_dir)
                draw_plot('lrs', trainer, output_dir=output_dir)
                with open(f'./{output_dir}/middle_logs.txt', 'a') as file:
                    file.write(f"\nEpoch {epoch} | Avg Loss: {epoch_loss/idx}\n")
                cleanup_memory()
                # 모델 저장 시 DDP 래핑된 경우, 내부 모듈(module)만 저장
                torch.save(model.module.state_dict(), f'./{output_dir}/model_{epoch}.pth')
        
        if dist.get_rank() == 0:
            scheduler.step()
            trainer['train_losses'].append(epoch_loss/len(train_loader))
            trainer['lrs'].append(optimizer.param_groups[0]['lr'])
            draw_plot('train_losses', trainer, output_dir=output_dir)
            draw_plot('lrs', trainer, output_dir=output_dir)
        
        if epoch % save_epoch == save_epoch - 1 and dist.get_rank() == 0:
            torch.save(model.module.state_dict(), f'./{output_dir}/model_{epoch}.pth')
        
        del loss, noised_inputs, output
        cleanup_memory()
        
        # ------------------------------
        # 검증 및 샘플 생성 (rank 0에서만 실행)
        # ------------------------------
        sfx_valid_loss = None
        model.eval()
        if dist.get_rank() == 0 and valid_loader is not None:
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
                device=device,
                clap_model=clap_model
            )
        
        if dist.get_rank() == 0:
            for vidx, d in enumerate(audios):
                ap = d['audio_path']
                prompt = d['prompt']
                duration = d['duration']
                
                audio_np, sr = librosa.load(ap, sr=44100)
                audio_tensor = torch.tensor(audio_np).to(device)
                audio_tensor = audio_tensor[:44100*10]
                audio_embed = clap_model(audio_tensor.unsqueeze(dim=0))
                melody = make_voice_cond(audio_tensor.cpu())
                zeros = torch.zeros((16, 215 - melody.shape[-1]))
                melody = torch.cat([melody, zeros], dim=-1)
                melody = melody.unsqueeze(dim=0).to(device)
                
                output_audio = generation(
                    model=model.module,
                    ae=ae,
                    text_conditioner=text_conditioner,
                    text=prompt,
                    steps=sample_cfg['sample_steps'],
                    cfg_scale=sample_cfg['sample_cfg'],
                    duration=duration,
                    sample_rate=sample_cfg['sample_rate'],
                    batch_size=1,
                    device=device,
                    disable=True,
                    melody_prompt=melody,
                    audio_embed=audio_embed
                )
                torchaudio.save(f'{output_dir}/generated_{epoch}_{vidx}.wav', output_audio.cpu()[0], 44100)
            
            if sfx_valid_loss is not None:
                trainer['valid_losses'].append(sfx_valid_loss)
                draw_plot('valid_losses', trainer, output_dir=output_dir)
            
            with open(f'./{output_dir}/logs.txt', 'a') as file:
                file.write(f"\nEpoch {epoch}\n")
                file.write(f"Train loss : {epoch_loss/len(train_loader)}\n")
                file.write(f"Valid loss : {sfx_valid_loss}\n")
        
        cleanup_memory()
    
    if dist.get_rank() == 0:
        print("Training finished")
    cleanup_distributed()
