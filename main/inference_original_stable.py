import os
import torch
import torchaudio
from dit_pure import DiffusionTransformer
from condition.t5condition import T5Conditioner
from inference.pure import generation
from config.model.config import config
from vae.get_function import create_autoencoder_from_config
from train.utils import cleanup_memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda'
model_config = config['model']

if __name__ == "__main__":
    print("Start inference")
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
        use_skip         = False
    )
    model = model.to(device)
    pre_model_dir = '/home/khj6051/mel_con_sample'
    # model.load_state_dict(torch.load(f'{pre_model_dir}/total_model_350.pth'))
    model.load_state_dict(torch.load('./stable_audio_origin_weight.pth'))

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n num_trainable_params : ", num_trainable_params, "\n")


    ae = create_autoencoder_from_config(config['auto_encoder_config']).to(device)
    ae_state_dict = torch.load(f'{pre_model_dir}/vae_weight.pth')
    ae_clean_state_dict = {layer_name.replace('model.', ''): weights for layer_name, weights in ae_state_dict.items()}
    ae.load_state_dict(ae_clean_state_dict)
    
    del ae_clean_state_dict
    del ae_state_dict
    cleanup_memory()

    text_conditioner = T5Conditioner(output_dim=768).to(device)
    # model.load_state_dict(torch.load(f'{pre_model_dir}/total_model_350.pth'), strict=False)

    ae.eval()
    text_conditioner.eval()
    model.eval()
    
    sample_cfg = {
        "sample_steps": 100,
        "sample_cfg": 6.5,
        "sample_duration": 30.0,
        "sample_rate": 44100,
        "diffusion_objective": "v"
    }

    print("start generation")

    for vidx, prompt in enumerate([
        # 'prompt: drum, hiphop, bpm: 100, key: D',
        # 'prompt: piano, hiphop, bpm: 100, key: D',
        # 'prompt: electric guitar, hiphop, bpm: 100, key: D',
        'Gentle drum rhythm perfect for initiating a video, accompanied by music.',
        'Soothing music evoking the warmth of hugging your father.',
        'hiphop style fast beat',
        'Rock beat played in a treated studio, session drumming on an acoustic kit',
        'Rock beat played in a treated studio, session drumming on an acoustic kit',
        'Rock beat played in a treated studio, session drumming on an acoustic kit',
        'Rock beat played in a treated studio, session drumming on an acoustic kit'
    ]):
        output = generation(
            model,
            ae,
            text_conditioner,
            text=prompt,
            steps=sample_cfg['sample_steps'],
            cfg_scale=sample_cfg['sample_cfg'],
            duration=sample_cfg['sample_duration'],
            sample_rate=sample_cfg['sample_rate'],
            batch_size=1,
            device=device,
            disable=False
        )
        print('index : ', vidx)
        torchaudio.save(f'./generated_{vidx}.wav', output.cpu()[0], sample_rate=44100)
    
    cleanup_memory()


# from train.dataset import SampleDataset
# from train.dataset import Config
# import torchaudio

# config = Config(
#     train_dataset='/home/khj6051/train_mix.csv', 
#     valid_dataset='/home/khj6051/valid_mix.csv', 
#     duration=10.0
# )

# print(config)

# valid_dataset = SampleDataset(config, mode="train", sample_rate=44100, force_channels='stereo')
# print(valid_dataset[0][0].shape, valid_dataset[0][1]['mixed_audio'].shape)
# # print(valid_dataset[1][0].shape, valid_dataset[1][1]['mixed_audio'].shape)
# # print(valid_dataset[2][0].shape, valid_dataset[2][1]['mixed_audio'].shape)

# for idx in [111, 2222, 3333, 34567, 44444]:
#     print(valid_dataset[idx][1]['caption'])
#     try:
#         torchaudio.save(f'./output/generated_{idx}_target.wav', valid_dataset[idx][0], sample_rate=44100)
#         torchaudio.save(f'./output/generated_{idx}_mixed.wav', valid_dataset[idx][1]['mixed_audio'], sample_rate=44100)
#     except Exception as e:
#         print(e)