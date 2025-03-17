import os
import math
import random
import pandas as pd
from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import ast

class Mono(nn.Module):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.mean(signal, dim=0, keepdim=True) if signal.ndim > 1 else signal

class Stereo(nn.Module):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim == 1:  # Mono to stereo
            return signal.unsqueeze(0).repeat(2, 1)
        elif signal.ndim == 2:
            if signal.shape[0] == 1:  # Mono to stereo
                return signal.repeat(2, 1)
            elif signal.shape[0] > 2:  # Trim to first two channels if more than two
                return signal[:2, :]
        return signal

class PadCrop_Normalized_T(nn.Module):
    def __init__(self, n_samples: int, sample_rate: int):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate

    def forward(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int, torch.Tensor]:
        n_channels, n_samples = source.shape
        
        if n_samples >= self.n_samples:
            # 입력이 길면 앞부분만 자름
            chunk = source[:, :self.n_samples]
        else:
            # 입력이 짧으면 반복하여 채우고 초과 부분 자름
            repeat_factor = (self.n_samples // n_samples) + 1
            extended_source = source.repeat(1, repeat_factor)  # 반복하여 확장
            chunk = extended_source[:, :self.n_samples]  # 필요한 길이만큼 자르기
        
        # Calculate times and padding mask
        t_start, t_end = 0.0, min(n_samples, self.n_samples) / self.n_samples
        seconds_start, seconds_total = 0, math.ceil(n_samples / self.sample_rate)
        padding_mask = torch.ones(self.n_samples)
        
        return chunk, t_start, t_end, seconds_start, seconds_total, padding_mask
        
class SampleDataset(Dataset):
    def __init__(self, config, mode: str = "train", sample_rate: int = 44100, force_channels: str = "stereo", transform=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.mode = mode
        self.transform = transform
        self.train_duration = config.train_duration
        self.group_size = 10
        
        if mode == "train":
            self.sample_size = int(config.train_duration * sample_rate)
            self.random_crop = True
        else:
            self.sample_size = int(config.valid_test_duration * sample_rate)
            self.random_crop = False

        self.encoding = torch.nn.Sequential(
            Stereo() if force_channels == "stereo" else torch.nn.Identity(),
            Mono() if force_channels == "mono" else torch.nn.Identity(),
        )

        self.pad_crop = PadCrop_Normalized_T(self.sample_size, self.sample_rate)

        csv_file = config.train_dataset if mode == "train" else config.valid_dataset
        self.data = pd.read_csv(csv_file)
        if mode != "train":
            self.data = self.data[:10000] # valid의 갯수를 만개만 쓰기
        print(f'Loaded {len(self.data)} samples for {mode} mode from {csv_file}')

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(filename)

    def load_file_segment(self, filename: str, start_sample: int, num_samples: int) -> torch.Tensor:
        try:
            audio, sr = torchaudio.load(filename, frame_offset=start_sample, num_frames=num_samples)
            audio = audio.float()
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(audio)
            if audio.numel() == 0:
                raise ValueError(f"Loaded audio data from {filename} is empty.")
            return audio
        except Exception as e:
            raise RuntimeError(f"Error loading audio segment from {filename}: {e}")

    def __len__(self) -> int:
        return len(self.data) // self.group_size

    def load_data(self, idx: int):
        row = self.data.iloc[idx]
        audio_filename = row['audio_path']

        if not self.file_exists(audio_filename):
            print(f"File does not exist: {audio_filename}")
            return self._load_data[random.randrange(len(self))]

        try:
            ti = torchaudio.info(audio_filename)
            original_sample_rate = ti.sample_rate
            duration = ti.num_frames/original_sample_rate
            total_samples = int(ti.num_frames)
    
            target_num_samples = int(self.sample_size * (original_sample_rate / self.sample_rate)) # 3*44100 * (48000/44100) = 144000
            
            # if self.random_crop and duration > self.train_duration:
            #     start_sample = random.randint(0, total_samples - target_num_samples)
            # else:
            #     start_sample = 0
            
            start_sample = 0
            # Load and process the audio segment
            audio = self.load_file_segment(audio_filename, start_sample, target_num_samples)
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
            audio = self.encoding(audio)
            audio = audio.clamp(-1, 1)
            
            if abs(torch.sum(audio))<0.000001 and duration<3:
                return self[random.randrange(len(self))]
            if abs(torch.sum(audio))<0.00001 and duration>=3:
                return self[random.randrange(len(self))]

            all_others = row['others']
            all_others = ast.literal_eval(all_others)

            mixed_audio = audio.clone()
            all_duration = [duration]
            for added_data in all_others:
                audio_path = added_data[0]
                duration = added_data[1]

                added_audio = self.load_file_segment(audio_path, start_sample, target_num_samples)
                added_audio, _, _, _, _, _ = self.pad_crop(added_audio)
                added_audio = self.encoding(added_audio)
                added_audio = added_audio.clamp(-1, 1)
                
                mixed_audio += added_audio
                all_duration.append(duration)

            caption = ', '.join(ast.literal_eval(row['text'])) if not pd.isna(row['text']) else ''
            
            max_duration = max(all_duration) if max(all_duration) >= 10.0 else max(all_duration)*2

            info = {
                "mixed_audio": mixed_audio,
                "audio_path": audio_filename,
                "seconds_start": start_sample / self.sample_rate,
                "seconds_total": max_duration,
                "duration": total_samples / self.sample_rate,
                # "padding_mask": padding_mask,
                "caption": caption
            }

            return audio, info

        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self._load_data[random.randrange(len(self))]
    
    def __getitem__(self, idx: int):
        """N개씩 묶어서 반환"""
        start_idx = idx * self.group_size
        end_idx = min(start_idx + self.group_size, len(self) * self.group_size)

        batch_audio = []
        batch_info = []

        for i in range(start_idx, end_idx):
            audio, info = self.load_data(i)  # 개별 샘플 로드
            batch_audio.append(audio)
            batch_info.append(info)
        
        return batch_audio, batch_info  # 리스트 형태로 반환

def add_random_noise(audio: torch.Tensor, noise_level: float = 0.0001) -> torch.Tensor:
    noise = torch.randn_like(audio) * noise_level
    return audio + noise

def time_shift(audio: torch.Tensor, shift_limit: int) -> torch.Tensor:
    shift = random.randint(-shift_limit, shift_limit)
    if shift == 0:
        return audio
    elif shift > 0:
        return torch.cat([audio[:, :, shift:], torch.zeros_like(audio[:, :, :shift])], dim=1)
    else:
        return torch.cat([torch.zeros_like(audio[:, :, :abs(shift)]), audio[:, :, :shift]], dim=1)

class Config:
    def __init__(self, train_dataset: str, valid_dataset: str, duration: float):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.duration = duration
        self.train_duration = duration
        self.valid_test_duration = duration
        self.dataset_chunk_size = None