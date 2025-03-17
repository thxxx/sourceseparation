from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int = 64
    gradient_clip_val: float = 0.2
    lr: float = 0.0001
    mlflow: bool = False
    num_workers: int = 4
    optimizer: str = "AdamW"
    output_dir: str = 'weights_new'
    scheduler: str = "CosineAnnealingWarmupRestarts"
    precision: str = "16-mixed"
    pretrain: bool = True
    project: str = "dit_voicesfx"
    sample_duration: float = 4.0
    sample_steps: int = 100
    sample_cfg: int = 6
    sample_rate: int = 44100
    is_make_sample: bool = True
    diffusion_objective: str = "v"
    betas = (0.9, 0.999)
    weight_decay: float = 0.001
