import os
import gc
import io
import base64
import random
import librosa
import torch
import torchaudio
import librosa.display
import numpy as np
import soundfile as sf
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce

def save_concatenated_mel_spectrogram(audio1, audio2, sr=44100, save_path='concatenated_mel_spectrogram.png', is_show=False):
    # Mel spectrogram 계산
    mel_spec1 = librosa.feature.melspectrogram(y=audio1, sr=sr, n_mels=128, fmax=8000)
    mel_spec2 = librosa.feature.melspectrogram(y=audio2, sr=sr, n_mels=128, fmax=8000)
    
    # Mel spectrogram을 dB 스케일로 변환
    mel_spec1_db = librosa.power_to_db(mel_spec1, ref=np.max)
    mel_spec2_db = librosa.power_to_db(mel_spec2, ref=np.max)
    # gap = np.full((mel_spec1_db.shape[0], 20), mel_spec1_db.min())  # 최소값으로 채운 간격 생성 (dB 스케일)
    
    # 두 Mel spectrogram을 가로로 이어 붙임
    # concatenated_mel_spec = np.hstack((mel_spec1_db, gap, mel_spec2_db))
    concatenated_mel_spec = np.hstack((mel_spec1_db, mel_spec2_db))
    
    # 그림 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(concatenated_mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Concatenated Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    if is_show:
        plt.show()
    plt.close()

def write_html(captions: list[str], audio_paths: list[Path], image_paths: list[Path]):
    html = """
    <html>
    <head>
        <title>Audio and Mel Preview</title>
    </head>
    <body>
        <table border="1">
            <tr>
                <th>Audio</th>
                <th>Mel</th>
            </tr>
    """

    # names = ["real", "pred", "gen"]
    for row_name, audio_path, image_path in zip(captions, audio_paths, image_paths):
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        html += f"""
            <tr>
                <td>
                    <p>{row_name}</p>
                    <audio controls>
                        <source src="data:audio/flac;base64,{audio_base64}" type="audio/flac">
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <img src="data:image/png;base64,{image_base64}" alt="{row_name} Mel Spectrogram" style="width:100%;">
                </td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    return html

def make_html(epoch, output_dir, data):
    # 저장 디렉토리 생성
    audio_dir = f'./{output_dir}/epoch_{epoch}/audio_files'
    spectrogram_dir = f'./{output_dir}/epoch_{epoch}/spectrogram_images'
    total_dir = f'./{output_dir}/generations'
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)

    audio_paths = []
    image_paths = []
    captions    = []
    sampling_rate = 44100
    n_mels        = 128  # Mel 필터의 개수
    hop_length    = 512  # Mel-spectrogram의 해상도를 결정하는 파라미터
    
    for i, item in enumerate(data):
        caption = item['caption']
        audio_array = item['array']
        
        # 오디오 파일로 저장
        audio_path = f"{audio_dir}/audio_{i}.wav"
        torchaudio.save(audio_path, audio_array, sampling_rate)
        audio_paths.append(audio_path)

        # Mel-spectrogram 생성
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_array.numpy()[0], sr=sampling_rate, n_mels=n_mels, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        spectrogram_path = f"{spectrogram_dir}/spectrogram_{i}.png"
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        image_paths.append(spectrogram_path)
        captions.append(caption)
        
    # HTML 코드 종료 부분
    html_content = write_html(captions, audio_paths, image_paths)
    
    # HTML 파일 저장
    with open(f"{total_dir}/audio_spectrogram_{epoch}.html", "w", encoding="utf-8") as file:
        file.write(html_content)


def draw_plot(key, trainer, output_dir):
    plt.figure(figsize=(10, 6))
    plt.title(f"{key}")
    plt.plot(trainer[key])
    plt.savefig(f'./{output_dir}/{key}.png')
    plt.close()

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

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

def calculate_targets(noise, z_0, alphas, sigmas, objective):
    if objective == "v":
        return noise * alphas - z_0 * sigmas
    elif objective == 'rectified_flow':
        return noise - z_0
    raise ValueError(f"Unknown objective: {objective}")

def prepare_batch_data(audio, info, ae, text_conditioner, device):
    with torch.no_grad():
        audio = audio.to(device)
        z_0 = ae.encode_audio(audio) # B, C, SL

        ma = info['mixed_audio'].to(device)
        mixed_z_0 = ae.encode_audio(ma) # B, C, SL
        
        text = info['caption']
        input_ids, attention_mask = text_conditioner(text, device=device)
        valid_tokens = attention_mask.sum(dim=1, keepdim=True)
        mean_pooled = input_ids.sum(dim=1)/valid_tokens

    return z_0, mixed_z_0, input_ids, attention_mask, mean_pooled

def prepare_batch_data_with_cross(audio, info, ae, text_conditioner, device, clap_model):
    with torch.no_grad():
        audio = audio.to(device)
        z_0 = ae.encode_audio(audio) # B, C, SL
        
        text = info['caption']
        input_ids, attention_mask = text_conditioner(text, device=device)
        valid_tokens = attention_mask.sum(dim=1, keepdim=True)
        mean_pooled = input_ids.sum(dim=1)/valid_tokens

        # print('assa', audio.shape)
        audb = clap_model(audio[:, 0, :])

    return z_0, input_ids, attention_mask, mean_pooled, audb

def make_xctx(z_0, seq_per_sec, seconds_total, p_uncond):
    bs, ch, seq_len = z_0.shape
    
    mask = torch.ones((bs, seq_len))
    
    voice_prompt = []
    for i in range(bs):
        duration = seconds_total[i]
        length = min(int(duration * seq_per_sec), seq_len)
        if random.random()>p_uncond and length>20:
            mask_length = torch.randint(int(length * 0.15), int(length * 0.3), (1,))
            mask[i, :mask_length] = 0
        else:
            mask_length = 0
        voice_prompt.append(
            torch.concat((z_0[i, :, :mask_length], torch.zeros((ch, seq_len - mask_length), device=device)), dim=-1)
        )
    
    voice_prompt = torch.stack(voice_prompt)

    return voice_prompt

def make_masked_audio(z_0, duration, p_uncond=0.3, masked_ratios=(0.5, 0.99)):
    bs, ch, sl = z_0.shape
    if random.random() < p_uncond or duration < 1.0:
        init_audio = torch.zeros_like(z_0).to(z_0.device)
        mask = torch.ones((bs, sl)).to(z_0.device).bool()
        return init_audio, mask
    
    masked = torch.zeros((bs, sl)).to(z_0.device)
    # 지워진 부분이 1, 나머지 부분이 0인 mask 만들기. 1인 부분의 loss를 계산하게 됨.
    sample_len = min(int(21.5 * duration)+1, 215)
    masked_ratio = random.uniform(masked_ratios[0], masked_ratios[-1])
    masked_len = int(sample_len * masked_ratio)
    start_idx = int(random.uniform(0, sample_len - masked_len))
    zeros = torch.zeros((bs, ch, masked_len)).to(z_0.device)
    zeros_pad = torch.zeros((bs, ch, 215-sample_len)).to(z_0.device)
    # init_audio = torch.concat((z_0[:, :, :start_idx], zeros, z_0[:, :, start_idx+masked_len:]), dim=-1)
    init_audio = torch.concat((z_0[:, :, :start_idx], zeros, z_0[:, :, start_idx+masked_len:sample_len], zeros_pad), dim=-1)
    
    masked[:, start_idx: start_idx+masked_len] = 1
    masked[:, sample_len+1:] = 1
    
    return init_audio, masked.bool()