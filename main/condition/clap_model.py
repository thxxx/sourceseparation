import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import condition.CLAP.src.laion_clap as laion_clap
from condition.CLAP.src.laion_clap import clap_module
from collections import OrderedDict

class CLAPModule(nn.Module):
    def __init__(self, model_output_dim=512, num_classes=None):
        super(CLAPModule, self).__init__()

        model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device='cuda')
        ckpt = torch.load('/workspace/music_audioset_epoch_15_esc_90.14.pt')['state_dict']
        # ckpt = torch.load('/workspace/mel_con_sample/main/condition/music_audioset_epoch_15_esc_90.14.pt')['state_dict']
        
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')  # 'module.' 제거
            new_state_dict[new_key] = value
        
        model.model.load_state_dict(new_state_dict, strict=False)
        clap_module = model.model
        del model
        del ckpt
        torch.cuda.empty_cache()
        
        self.module = clap_module

        # self.proj_out = nn.Sequential(
        #     nn.Linear(model_output_dim, model_output_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(model_output_dim*2, 1)
        # )
    
    def forward(self, x):
        bs = x.shape[0]
        input_dict_list = [{'waveform': d} for d in x]
        out = self.module.get_audio_embedding(input_dict_list)
        
        # out = self.proj_out(out)
        return out