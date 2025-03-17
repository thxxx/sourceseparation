config = {
    'auto_encoder_config': {
        'encoder': {
            'type': 'oobleck', # 현재는 oobleck 혹은 dac만 가능
            'requires_grad': False,
            'config': {
                'in_channels': 2,
                'channels': 128,
                'c_mults': [1, 2, 4, 8, 16],
                'strides': [2, 4, 4, 8, 8],
                'latent_dim': 128,
                'use_snake': True
            }
        },
        'decoder': {
            'type': 'oobleck',
            'requires_grad': False,
            'config': {
                'out_channels': 2,
                'channels': 128,
                'c_mults': [1, 2, 4, 8, 16],
                'strides': [2, 4, 4, 8, 8],
                'latent_dim': 64,
                'use_snake': True,
                'final_tanh': False
            }
        },
        'bottleneck': {'type': 'vae'},
        'latent_dim': 64,
        'downsampling_ratio': 2048,
        'io_channels': 2,
        'sample_rate': 44100
    },
    "text_conditioner_config": {"output_dim": 768, 't5_model_name': 't5-base', 'max_length': 128},
    "timing_config": {"output_dim": 768, 'min_val': 0, 'max_val': 512},
    "model": {
        "cond_embed_dim": 768,
        "ada_cond_dim": None,
    },
    "cond_token_dim": 768,
    "project_cond_tokens": False,
    "global_cond_dim": 1536,
    "cross_attn_cond_keys": ['prompt', 'seconds_start', 'seconds_total'],
    "global_cond_keys": ['seconds_start', 'seconds_total']
}
