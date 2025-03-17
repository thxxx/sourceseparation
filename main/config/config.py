from dataclasses import dataclass, field

from omegaconf import MISSING, DictConfig

from config.data.config import DataConfig
from config.train.config import TrainConfig

defaults = ["_self_", {"model": "dit"}, {"data": "base"}, {"train": "base"}]

@dataclass
class Config:
    defaults: list = field(default_factory=lambda: defaults)

    # model: AudioBoxBaseConfig = MISSING
    data: DataConfig = MISSING
    train: TrainConfig = MISSING

def dict_to_config(cfg: dict | DictConfig):
    model_config = AudioBoxBaseConfig(**cfg["model"])
    data_config = DataConfig(**cfg["data"])
    train_config = TrainConfig(**cfg["train"])
    
    return Config(
        model=model_config,
        data=data_config,
        train=train_config,
    )